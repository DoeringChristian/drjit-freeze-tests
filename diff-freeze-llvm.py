import drjit as dr
import mitsuba as mi
import time

mi.set_variant("llvm_ad_rgb")

key = "red.reflectance.value"


def mse(image, image_ref):
    return dr.mean(dr.mean(dr.square(image - image_ref)))


def optimize(scene, opt, ref, other):
    opt[key] = other[0]

    params = mi.traverse(scene)
    params.update(opt)

    image = mi.render(scene, params, spp=4)

    # Evaluate the objective function from the current rendered image
    loss = mse(image, ref)

    # Backpropagate through the rendering process
    dr.backward(loss)

    # Optimizer: take a gradient descent step
    opt.step()

    # Post-process the optimized parameters to ensure legal color values.
    opt[key] = dr.clamp(opt[key], 0.0, 1.0)

    other[0] = opt[key]

    return image


if __name__ == "__main__":
    # dr.set_log_level(dr.LogLevel.Trace)
    # dr.set_flag(dr.JitFlag.Debug, True)
    dr.set_flag(dr.JitFlag.ReuseIndices, False)
    dr.set_flag(dr.JitFlag.LaunchBlocking, True)
    dr.set_flag(dr.JitFlag.OptimizeLoops, True)

    iteration_count = 50
    burnin = 3

    # Frozen:
    frozen = dr.freeze(optimize)

    scene = mi.cornell_box()
    scene["integrator"] = {
        "type": "prb",
    }
    scene = mi.load_dict(scene)
    params = mi.traverse(scene)
    params["light.emitter.radiance.value"] = mi.Color3f(30, 30, 30)
    params.update()

    image_ref = mi.render(scene, spp=512)

    # Preview the reference image
    mi.util.write_bitmap("out/ref.jpg", image_ref)

    # Save the original value
    param_ref = mi.Color3f(params[key])

    # Set another color value and update the scene
    params[key] = mi.Color3f(0.01, 0.2, 0.9)
    params.update()

    image_init = mi.render(scene, params, spp=128)
    mi.util.write_bitmap("out/init.jpg", image_init)

    opt = mi.ad.Adam(lr=0.05)
    opt[key] = params[key]
    params.update(opt)

    t_frozen = 0
    errors = []
    for it in range(iteration_count):
        start = time.time()
        dr.sync_thread()
        image = frozen(scene, opt, image_ref, [opt[key], opt.lr_default_v, opt.state])
        dr.sync_thread()
        end = time.time()

        # Track the difference between the current color and the true value
        err_ref = dr.sum(dr.sqr(param_ref - params[key]))
        print(f"Iteration {it:02d}: parameter error = {err_ref[0]:6f}")
        errors.append(err_ref)

        mi.util.write_bitmap(f"out/{it}.jpg", image)

        if it >= burnin:
            t_frozen += end - start
    print("\nOptimization complete.")

    image_final = mi.render(scene, spp=128)
    mi.util.write_bitmap("out/final.jpg", image_final)
