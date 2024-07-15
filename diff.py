import mitsuba as mi
import drjit as dr
import numpy as np

mi.set_variant("cuda_ad_rgb")

if __name__ == "__main__":
    dr.set_log_level(dr.LogLevel.Trace)
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    dr.set_flag(dr.JitFlag.VCallOptimize, True)
    dr.set_flag(dr.JitFlag.SymbolicCalls, True)
    dr.set_flag(dr.JitFlag.Debug, True)
    dr.set_flag(dr.JitFlag.ReuseIndices, False)
    dr.set_flag(dr.JitFlag.LaunchBlocking, True)
    # dr.set_log_level(dr.LogLevel.Debug)
    scene = mi.cornell_box()
    w = 128
    h = 128
    scene["sensor"]["film"]["width"] = w
    scene["sensor"]["film"]["height"] = h

    scene: mi.Scene = mi.load_dict(scene)

    img_ref = mi.render(scene, spp=1)
    mi.util.write_bitmap("out/ref.jpg", img_ref)

    params = mi.traverse(scene)
    key = "red.reflectance.value"

    params[key] = mi.Color3f(0.01, 0.2, 0.9)
    dr.make_opaque(params[key])
    params.update()
    opt = mi.ad.Adam(lr=0.05)
    opt[key] = params[key]
    print(f"{dr.grad_enabled(opt[key])=}")
    params.update(opt)

    # print(f"{dir(dr)=}")

    for it in range(10):
        img = mi.render(scene, params, spp=1)

        loss = dr.mean(dr.square(img - img_ref).array)
        print(f"{it=}, {loss=}")

        dr.backward(loss)

        opt.step()

        opt[key] = dr.clamp(opt[key], 0.0, 1.0)

        params.update(opt)

        mi.util.write_bitmap(f"out/{it}.jpg", img)
