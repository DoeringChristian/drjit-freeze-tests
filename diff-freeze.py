import drjit as dr
import mitsuba as mi
import time

mi.set_variant("cuda_ad_rgb")


def mse(image, image_ref):
    return dr.mean(dr.mean(dr.square(image - image_ref)))


def optimize(scene, opt, ref, params):
    params = mi.traverse(scene)
    # Perform a (noisy) differentiable rendering of the scene
    image = mi.render(scene, params, spp=4)

    # Evaluate the objective function from the current rendered image
    loss = mse(image, ref)

    # Backpropagate through the rendering process
    dr.backward(loss)

    # Optimizer: take a gradient descent step
    opt.step()

    # Post-process the optimized parameters to ensure legal color values.
    opt[key] = dr.clamp(opt[key], 0.0, 1.0)


if __name__ == "__main__":
    # dr.set_log_level(dr.LogLevel.Trace)
    # dr.set_flag(dr.JitFlag.Debug, True)
    dr.set_flag(dr.JitFlag.ReuseIndices, False)
    dr.set_flag(dr.JitFlag.LaunchBlocking, True)
    dr.set_flag(dr.JitFlag.OptimizeLoops, True)

    iteration_count = 50
    burnin = 3

    scene = mi.cornell_box()
    scene["integrator"] = {
        "type": "prb",
    }
    scene = mi.load_dict(scene)
    image_ref = mi.render(scene, spp=512)

    # Preview the reference image
    mi.util.write_bitmap("out/ref.jpg", image_ref)

    params = mi.traverse(scene)

    key = "red.reflectance.value"

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

    errors = []
    t_ref = 0
    t_exec = 0
    # for it in range(iteration_count):
    #     start = time.time()
    #     dr.sync_thread()
    #     optimize(scene, opt, image_ref)
    #     dr.sync_thread()
    #     end = time.time()
    #
    #     # Update the scene state to the new optimized values
    #     params.update(opt)
    #
    #     # Track the difference between the current color and the true value
    #     err_ref = dr.sum(dr.sqr(param_ref - params[key]))
    #     print(f"Iteration {it:02d}: parameter error = {err_ref[0]:6f}", end="\r")
    #     errors.append(err_ref)
    #
    #     history = dr.kernel_history()
    #     execution_time = sum(
    #         [k["execution_time"] for k in history if "execution_time" in k]
    #     )
    #
    #     if it >= burnin:
    #         t_ref += end - start
    #         t_exec += execution_time
    # print("\nOptimization complete.")

    image_final = mi.render(scene, spp=128)
    mi.util.write_bitmap("out/final.jpg", image_final)

    # Frozen:
    frozen = dr.freeze(optimize)

    scene = mi.cornell_box()
    scene["integrator"] = {
        "type": "prb",
    }
    scene = mi.load_dict(scene)
    image_ref = mi.render(scene, spp=512)

    # Preview the reference image
    mi.util.write_bitmap("out/ref.jpg", image_ref)

    params = mi.traverse(scene)

    key = "red.reflectance.value"

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
        frozen(scene, opt, image_ref, (opt[key], opt.lr_default_v))
        dr.sync_thread()
        end = time.time()

        # Update the scene state to the new optimized values
        params.update(opt)

        # Track the difference between the current color and the true value
        err_ref = dr.sum(dr.sqr(param_ref - params[key]))
        print(f"Iteration {it:02d}: parameter error = {err_ref[0]:6f}")
        errors.append(err_ref)

        if it >= burnin:
            t_frozen += end - start
    print("\nOptimization complete.")

    image_final = mi.render(scene, spp=128)
    mi.util.write_bitmap("out/final.jpg", image_final)

    t_exec /= 1000
    n = iteration_count - burnin
    print(f"avg: {t_ref/n=}s, {t_exec/n=}s, {t_frozen/n=}")
