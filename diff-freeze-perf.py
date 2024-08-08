import drjit as dr
import mitsuba as mi
import time

mi.set_variant("cuda_ad_rgb")

key = "red.reflectance.value"


def mse(image, image_ref, n):
    return dr.sum(dr.square(image - image_ref), axis=None) / n


def optimize(scene, opt, ref, n, other):
    opt[key] = other[0]

    params = mi.traverse(scene)
    params.update(opt)

    image = mi.render(scene, params, spp=1)

    # Evaluate the objective function from the current rendered image
    loss = mse(image, ref, n)

    # Backpropagate through the rendering process
    dr.backward(loss)

    # Optimizer: take a gradient descent step
    opt.step()

    # Post-process the optimized parameters to ensure legal color values.
    opt[key] = dr.clamp(opt[key], 0.0, 1.0)

    other[0] = opt[key]

    return image, loss


def run(optimize, b, n, name):

    scene = mi.cornell_box()
    scene["integrator"] = {
        "type": "prb",
    }
    print(f"{scene=}")
    w = 128
    h = 128
    c = 3
    scene["sensor"]["film"]["width"] = w
    scene["sensor"]["film"]["height"] = h
    scene = mi.load_dict(scene)
    params = mi.traverse(scene)
    params["light.emitter.radiance.value"] = mi.Color3f(30, 30, 30)
    params.update()

    image_ref = mi.render(scene, spp=4)

    # Preview the reference image
    mi.util.write_bitmap(f"out/{name}/ref.jpg", image_ref)

    # Save the original value
    param_ref = mi.Color3f(params[key])

    opt = mi.ad.Adam(lr=0.05)
    opt[key] = mi.Color3f(0.01, 0.2, 0.9)
    params.update(opt)

    duration = 0
    for it in range(b + n):
        # opt[key] = mi.Color3f(0.01, 0.2, 0.9)
        print(f"before: {opt[key]=}, {params[key]=}")
        n = dr.opaque(mi.Float, w * h * c)
        start = time.time()
        dr.sync_thread()
        with dr.profile_range("optimize"):
            image, loss = optimize(
                scene, opt, image_ref, n, [opt[key], opt.lr_default_v, opt.state]
            )
        dr.sync_thread()
        end = time.time()

        mi.util.write_bitmap(f"out/{name}/{it}.jpg", image)

        print(f"after: {opt[key]=}, {params[key]=}, {loss=}")

        if it >= b:
            duration += end - start
    print("\nOptimization complete.")

    image_final = mi.render(scene, spp=4)
    mi.util.write_bitmap(f"out/{name}/final.jpg", image_final)

    return duration


if __name__ == "__main__":
    # dr.set_log_level(dr.LogLevel.Debug)
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
