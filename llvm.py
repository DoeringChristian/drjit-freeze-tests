import mitsuba as mi
import drjit as dr
import time
import numpy as np

# mi.set_variant("cuda_ad_rgb")
mi.set_variant("llvm_ad_rgb")

# import mypath

k = "light.emitter.radiance.value"

if __name__ == "__main__":
    # dr.set_log_level(dr.LogLevel.Debug)
    # dr.set_flag(dr.JitFlag.VCallOptimize, False)
    # dr.set_flag(dr.JitFlag.SymbolicCalls, True)
    # dr.set_flag(dr.JitFlag.Debug, True)

    # dr.set_log_level(dr.LogLevel.Trace)
    # dr.set_flag(dr.JitFlag.ReuseIndices, False)
    dr.set_flag(dr.JitFlag.LaunchBlocking, True)
    # dr.set_flag(dr.JitFlag.KernelHistory, True)

    def func(scene: mi.Scene) -> mi.TensorXf:
        dr.kernel_history_clear()
        with dr.profile_range("render"):
            result = mi.render(scene, spp=1)
            dr.eval(result)
        return result

    def run(scene: mi.Scene, f, n):
        # params = mi.traverse(scene)
        for i in range(n):
            # params[k].x = value + 10.0 * i
            # params.update()

            result = f(scene)
            dr.eval(result)

    w = 1024
    h = 1024

    scene = mi.cornell_box()
    # del scene["large-box"]
    # del scene["small-box"]
    scene["sensor"]["film"]["width"] = w
    scene["sensor"]["film"]["height"] = h

    scene = mi.load_dict(scene, parallel=False)

    params = mi.traverse(scene)
    # print(params)

    n = 10

    # value = mi.Float(params[k].x)

    run(scene, func, n)
    run(scene, dr.freeze(func), n)
