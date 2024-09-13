import mitsuba as mi
import drjit as dr
import time
import numpy as np

mi.set_variant("cuda_ad_rgb")

if __name__ == "__main__":
    # dr.set_log_level(dr.LogLevel.Trace)
    # dr.set_log_level(dr.LogLevel.Debug)
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    dr.set_flag(dr.JitFlag.VCallOptimize, False)
    dr.set_flag(dr.JitFlag.SymbolicCalls, True)
    # dr.set_flag(dr.JitFlag.Debug, True)
    dr.set_flag(dr.JitFlag.ReuseIndices, False)
    dr.set_flag(dr.JitFlag.LaunchBlocking, True)

    def func(x: mi.Float, scene: mi.Scene) -> mi.Float:
        return x + 1

    w = 128
    h = 128

    scene = mi.cornell_box()
    del scene["large-box"]
    del scene["small-box"]
    scene["sensor"]["film"]["width"] = w
    scene["sensor"]["film"]["height"] = h
    scene["white"] = {
        "type": "principled",
        "base_color": {
            "type": "rgb",
            "value": [0.9, 0.9, 0.9],
        },
    }
    scene = mi.load_dict(scene)
    print(f"{scene=}")
    params = mi.traverse(scene)
    print(params)

    b = 1  # burnin
    n = 4

    for i in range(n + b):

        x = dr.arange(mi.Float, 128)
        dr.make_opaque(x)

        dr.kernel_history_clear()

        dr.sync_thread()
        start = time.time()
        reference = func(x, scene)
        dr.eval(reference)
        dr.sync_thread()
        end = time.time()

        history = dr.kernel_history()
        execution_time = sum(
            [k["execution_time"] for k in history if "execution_time" in k]
        )
        print(
            f"reference total: {end - start}s, execution_time: {execution_time/1000}s"
        )

    frozen = dr.freeze(func)

    for i in range(n + b):

        x = dr.arange(mi.Float, 128)
        dr.make_opaque(x)

        dr.sync_thread()
        start = time.time()
        result = frozen(x, scene)
        dr.eval(result)
        dr.sync_thread()
        end = time.time()

        print(f"frozen took: {(end - start)}s")
