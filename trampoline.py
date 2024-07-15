import mitsuba as mi
import drjit as dr
import numpy as np

mi.set_variant("cuda_ad_rgb")

if __name__ == "__main__":
    # dr.set_log_level(dr.LogLevel.Trace)
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    dr.set_flag(dr.JitFlag.VCallOptimize, True)
    dr.set_flag(dr.JitFlag.SymbolicCalls, True)
    dr.set_flag(dr.JitFlag.Debug, True)
    dr.set_flag(dr.JitFlag.ReuseIndices, False)
    dr.set_flag(dr.JitFlag.LaunchBlocking, True)

    integrator = mi.load_dict(
        {
            "type": "prb",
        }
    )
    print(f"{type(integrator)=}")
    # exit()

    def func(sensor: mi.Integrator) -> mi.UInt32:
        return dr.arange(mi.UInt32, 10)

    frozen = dr.freeze(func)

    for i in range(3):
        frozen(integrator)
