import mitsuba as mi
import drjit as dr
import time
import numpy as np

mi.set_variant("cuda_ad_rgb")
# mi.set_variant("llvm_ad_rgb")

# import mypath

if __name__ == "__main__":
    dr.set_log_level(dr.LogLevel.Trace)
    # dr.set_flag(dr.JitFlag.VCallOptimize, False)
    # dr.set_flag(dr.JitFlag.SymbolicCalls, True)
    # dr.set_flag(dr.JitFlag.Debug, True)

    # dr.set_log_level(dr.LogLevel.Trace)
    # dr.set_flag(dr.JitFlag.ReuseIndices, False)
    dr.set_flag(dr.JitFlag.LaunchBlocking, False)
    # dr.set_flag(dr.JitFlag.KernelHistory, True)

    def func(scene: mi.Scene) -> mi.TensorXf:
        with dr.profile_range("render"):
            result = mi.render(scene, spp=1)
            dr.eval(result)
        return result

    w = 128
    h = 128

    scene = mi.cornell_box()
    scene["sensor"]["film"]["width"] = w
    scene["sensor"]["film"]["height"] = h

    scene = mi.load_dict(scene, parallel=False)

    b = 2
    n = 10

    t_ref = 0
    t_ref_exec = 0

    for i in range(b + n):
        reference = func(scene)
        dr.eval(reference)

    frozen = dr.freeze(func)

    t_frozen = 0

    for i in range(b + n):
        result = frozen(scene)
        dr.eval(result)
