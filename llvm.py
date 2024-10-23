import mitsuba as mi
import drjit as dr
import time
import numpy as np
import tqdm

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
        result = mi.render(scene, spp=1)
        dr.eval(result)
        return result

    def run(scene: mi.Scene, f, b, n):
        duration = 0
        for i in tqdm.tqdm(range(b + n)):
            start = time.time()
            dr.sync_thread()

            with dr.profile_range("func"):
                result = f(scene)

            dr.sync_thread()
            end = time.time()
            if i > b:
                duration += end - start

        return duration / n

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

    b = 10
    n = 100

    # value = mi.Float(params[k].x)

    normal = run(scene, func, b, n)
    frozen = run(scene, dr.freeze(func), b, n)

    print(f"normal: {normal}s, frozen: {frozen}s")
