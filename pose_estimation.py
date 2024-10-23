import drjit as dr
import mitsuba as mi

mi.set_variant("llvm_ad_rgb")


def main():
    # dr.set_log_level(dr.LogLevel.Trace)
    # dr.set_flag(dr.JitFlag.ReuseIndices, False)
    # dr.set_flag(dr.JitFlag.Debug, True)
    # dr.set_flag(dr.JitFlag.LaunchBlocking, True)
    # dr.set_flag(dr.JitFlag.OptimizeCalls, False)
    w = 128
    h = 128

    def apply_transformation(initial_vertex_positions, opt, params):
        opt["trans"] = dr.clip(opt["trans"], -0.5, 0.5)
        opt["angle"] = dr.clip(opt["angle"], -0.5, 0.5)

        trafo = (
            mi.Transform4f()
            .translate([opt["trans"].x, opt["trans"].y, 0.0])
            .rotate([0, 1, 0], opt["angle"] * 100.0)
        )

        print("ravel:")
        params["bunny.vertex_positions"] = dr.ravel(trafo @ initial_vertex_positions)

    def mse(image, image_ref):
        return dr.sum(dr.square(image - image_ref), axis=None)

    def optimize(scene, ref, initial_vertex_positions, other):
        params = mi.traverse(scene)
        print(f"{dr.grad_enabled(params['bunny.vertex_positions'])=}")

        image = mi.render(scene, params, spp=1, seed=1, seed_grad=2)

        # Evaluate the objective function from the current rendered image
        loss = mse(image, ref)
        print(f"{type(loss)=}")

        # Backpropagate through the rendering process
        dr.backward(loss)

        return image, loss

    def load_scene():
        from mitsuba.scalar_rgb import Transform4f as T

        scene = mi.cornell_box()
        del scene["large-box"]
        del scene["small-box"]
        del scene["green-wall"]
        del scene["red-wall"]
        del scene["floor"]
        del scene["ceiling"]
        scene["bunny"] = {
            "type": "ply",
            "filename": "scenes/meshes/bunny.ply",
            "to_world": T().scale(6.5),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": (0.3, 0.3, 0.75)},
            },
        }
        scene["integrator"] = {
            "type": "prb",
        }
        scene["sensor"]["film"] = {
            "type": "hdrfilm",
            "width": w,
            "height": h,
            "rfilter": {"type": "gaussian"},
            "sample_border": True,
        }

        scene = mi.load_dict(scene, parallel=False)
        return scene

    def run(scene: mi.Scene, optimize, n) -> tuple[mi.TensorXf, mi.Point3f, mi.Float]:
        params = mi.traverse(scene)

        params.keep("bunny.vertex_positions")
        initial_vertex_positions = dr.unravel(
            mi.Point3f, params["bunny.vertex_positions"]
        )

        image_ref = mi.render(scene, spp=4)

        opt = mi.ad.Adam(lr=0.025)
        opt["angle"] = mi.Float(0.25)
        opt["trans"] = mi.Point3f(0.1, -0.25, 0.0)

        for i in range(n):
            params = mi.traverse(scene)
            params.keep("bunny.vertex_positions")

            apply_transformation(initial_vertex_positions, opt, params)
            print(f"{dr.grad_enabled(params['bunny.vertex_positions'])=}")

            with dr.profile_range("optimize"):
                image, loss = optimize(
                    scene,
                    image_ref,
                    initial_vertex_positions,
                    [
                        params["bunny.vertex_positions"],
                    ],
                )

            opt.step()

        image_final = mi.render(scene, spp=4, seed=1, seed_grad=2)

        return image_final, opt["trans"], opt["angle"]

    n = 10

    # NOTE:
    # In this cas, we have to use the same scene object
    # for the frozen and non-frozen case, as re-loading
    # the scene causes mitsuba to render different images,
    # leading to diverging descent traijectories.

    scene = load_scene()
    params = mi.traverse(scene)
    initial_vertex_positions = mi.Float(params["bunny.vertex_positions"])

    print("Reference:")
    img_ref, trans_ref, angle_ref = run(scene, optimize, n)

    # Reset parameters:
    params["bunny.vertex_positions"] = initial_vertex_positions
    params.update()

    print("Frozen:")
    img_frozen, trans_frozen, angle_frozen = run(scene, optimize, n)

    # NOTE: cannot compare results as errors accumulate and the result will never be the same.

    assert dr.allclose(trans_ref, trans_frozen)
    assert dr.allclose(angle_ref, angle_frozen)
    if integrator != "prb_projective":
        print(f"{dr.max(dr.abs(img_ref - img_frozen), axis=None)=}")
        assert dr.allclose(img_ref, img_frozen, atol=1e-4)


if __name__ == "__main__":
    main()
