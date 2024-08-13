import drjit as dr
import mitsuba as mi
import time

mi.set_variant("cuda_ad_rgb")


def mse(image, image_ref):
    return dr.mean(dr.square(image - image_ref), axis=None)


def apply_transformation(initial_vertex_positions, opt, params):
    opt["trans"] = dr.clamp(opt["trans"], -0.5, 0.5)
    opt["angle"] = dr.clamp(opt["angle"], -0.5, 0.5)

    trafo = (
        mi.Transform4f()
        .translate([opt["trans"].x, opt["trans"].y, 0.0])
        .rotate([0, 1, 0], opt["angle"] * 100.0)
    )

    # print(f"{dir(trafo.matrix)=}")
    # print(f"{trafo.matrix.index_ad=}")
    # dr.set_label(initial_vertex_positions, "initial_vertex_positions")
    # dr.set_label(trafo, "trafo")
    # dr.set_label(trafo.matrix, "trafo.matrix.array")
    # dr.set_label(opt["trans"].x, "trans.x")

    print("ravel:")
    params["bunny.vertex_positions"] = dr.ravel(opt["trans"] + initial_vertex_positions)


def optimize(scene, opt, ref, initial_vertex_positions, other):
    params = mi.traverse(scene)

    image = mi.render(scene, params, spp=1)

    # Evaluate the objective function from the current rendered image
    loss = mse(image, ref)
    print(f"{type(loss)=}")

    # Backpropagate through the rendering process
    dr.backward(loss)

    return image, loss


def run(optimize, b, n, name):
    from mitsuba.scalar_rgb import Transform4f as T

    w = 128
    h = 128
    c = 3

    scene = mi.cornell_box()
    del scene["large-box"]
    del scene["small-box"]
    del scene["green-wall"]
    del scene["red-wall"]
    del scene["floor"]
    del scene["ceiling"]
    scene["bunny"] = {
        "type": "ply",
        "filename": "scenes/cbox/meshes/bunny.ply",
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

    # scene = {
    #     "type": "scene",
    #     "integrator": {"type": "prb"},
    #     "sensor": {
    #         "type": "perspective",
    #         "to_world": T().look_at(origin=(0, 0, 2), target=(0, 0, 0), up=(0, 1, 0)),
    #         "fov": 60,
    #         "film": {
    #             "type": "hdrfilm",
    #             "width": w,
    #             "height": h,
    #             "rfilter": {"type": "gaussian"},
    #             "sample_border": True,
    #         },
    #     },
    #     "wall": {
    #         "type": "obj",
    #         "filename": "scenes/cbox/meshes/rectangle.obj",
    #         "to_world": T().translate([0, 0, -2]).scale(2.0),
    #         "face_normals": True,
    #         "bsdf": {
    #             "type": "diffuse",
    #             "reflectance": {"type": "rgb", "value": (0.5, 0.5, 0.5)},
    #         },
    #     },
    #     "bunny": {
    #         "type": "ply",
    #         "filename": "scenes/cbox/meshes/bunny.ply",
    #         "to_world": T().scale(6.5),
    #         "bsdf": {
    #             "type": "diffuse",
    #             "reflectance": {"type": "rgb", "value": (0.3, 0.3, 0.75)},
    #         },
    #     },
    #     "light": {
    #         "type": "obj",
    #         "filename": "scenes/cbox/meshes/sphere.obj",
    #         "emitter": {
    #             "type": "area",
    #             "radiance": {"type": "rgb", "value": [1e3, 1e3, 1e3]},
    #         },
    #         "to_world": T().translate([2.5, 2.5, 7.0]).scale(0.25),
    #     },
    # }

    # print(f"{scene=}")
    # exit()
    scene = mi.load_dict(scene)
    params = mi.traverse(scene)
    # print(params)
    # exit()
    params.keep("bunny.vertex_positions")
    initial_vertex_positions = dr.unravel(mi.Point3f, params["bunny.vertex_positions"])

    image_ref = mi.render(scene, spp=4)

    # Preview the reference image
    mi.util.write_bitmap(f"out/{name}/ref.jpg", image_ref)

    opt = mi.ad.Adam(lr=0.025)
    opt["angle"] = mi.Float(0.25)
    opt["trans"] = mi.Point3f(0.1, -0.25, 0.0)

    # apply_transformation(initial_vertex_positions, opt, params)
    # print("params.update()")
    # params.update(opt)
    # print("end params.update()")

    duration = 0
    for it in range(b + n):
        # opt[key] = mi.Color3f(0.01, 0.2, 0.9)
        print(f"before: {opt['trans']=}")
        n = dr.opaque(mi.Float, w * h * c)

        params = mi.traverse(scene)
        params.keep("bunny.vertex_positions")
        print(f"before: {dr.max(dr.grad(params['bunny.vertex_positions']))=}")
        apply_transformation(initial_vertex_positions, opt, params)
        print("params.update()")
        params.update(opt)
        print("end params.update()")

        start = time.time()
        dr.sync_thread()
        with dr.profile_range("optimize"):
            image, loss = optimize(
                scene,
                opt,
                image_ref,
                initial_vertex_positions,
                [
                    params["bunny.vertex_positions"],
                ],
            )
        dr.sync_thread()
        end = time.time()
        print(f"after: {dr.max(dr.grad(params['bunny.vertex_positions']))=}")
        dr.backward_from(params["bunny.vertex_positions"])

        opt.step()

        mi.util.write_bitmap(f"out/{name}/{it}.jpg", image)

        if it >= b:
            duration += end - start
    print("\nOptimization complete.")

    image_final = mi.render(scene, spp=4)
    mi.util.write_bitmap(f"out/{name}/final.jpg", image_final)

    return duration


if __name__ == "__main__":
    # dr.set_log_level(dr.LogLevel.Trace)
    # dr.set_flag(dr.JitFlag.Debug, True)
    dr.set_flag(dr.JitFlag.ReuseIndices, False)
    dr.set_flag(dr.JitFlag.LaunchBlocking, True)
    dr.set_flag(dr.JitFlag.OptimizeLoops, True)

    n = 2
    b = 4

    print("Reference:")
    t_ref = run(optimize, b, n, "ref")
    print("Frozen:")
    t_frozen = run(dr.freeze(optimize), b, n, "frozen")

    print(f"Duration avg: {t_ref/n=}, {t_frozen/n=}")
