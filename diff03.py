import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

if __name__ == "__main__":
    dr.set_log_level(dr.LogLevel.Trace)
    dr.set_flag(dr.JitFlag.Debug, True)
    dr.set_flag(dr.JitFlag.LoopOptimize, True)
    dr.set_flag(dr.JitFlag.ReuseIndices, False)
    dr.set_flag(dr.JitFlag.LaunchBlocking, True)
    scene = mi.cornell_box()
    scene["integrator"] = {
        "type": "prb",
    }
    del scene["small-box"]
    del scene["large-box"]
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

    image_init = mi.render(scene, spp=128)
    mi.util.write_bitmap("out/init.jpg", image_init)

    opt = mi.ad.Adam(lr=0.05)
    opt[key] = params[key]
    params.update(opt)

    def mse(image):
        return dr.mean(dr.mean(dr.square(image - image_ref)))

    iteration_count = 50

    errors = []
    for it in range(iteration_count):
        # Perform a (noisy) differentiable rendering of the scene
        image = mi.render(scene, spp=4)
        mi.util.write_bitmap(f"out/{it}.jpg", image)

        # Evaluate the objective function from the current rendered image
        loss = mse(image)

        # Backpropagate through the rendering process
        dr.backward(loss)

        # Optimizer: take a gradient descent step
        opt.step()

        # Post-process the optimized parameters to ensure legal color values.
        opt[key] = dr.clamp(opt[key], 0.0, 1.0)

        # Update the scene state to the new optimized values
        params.update(opt)

        # Track the difference between the current color and the true value
        err_ref = dr.sum(dr.sqr(param_ref - params[key]))
        print(f"Iteration {it:02d}: parameter error = {err_ref[0]:6f}", end="\r")
        errors.append(err_ref)
    print("\nOptimization complete.")

    image_final = mi.render(scene, spp=128)
    mi.util.write_bitmap("out/final.jpg", image_final)
