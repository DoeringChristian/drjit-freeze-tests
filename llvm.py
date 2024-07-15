import mitsuba as mi
import drjit as dr
import time
import numpy as np

# mi.set_variant("cuda_ad_rgb")
mi.set_variant("llvm_ad_rgb")

# import mypath

if __name__ == "__main__":
    dr.set_log_level(dr.LogLevel.Trace)
    # dr.set_log_level(dr.LogLevel.Debug)
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    dr.set_flag(dr.JitFlag.VCallOptimize, False)
    dr.set_flag(dr.JitFlag.SymbolicCalls, True)
    dr.set_flag(dr.JitFlag.Debug, True)
    dr.set_flag(dr.JitFlag.ReuseIndices, False)
    dr.set_flag(dr.JitFlag.LaunchBlocking, True)

    def func(scene: mi.Scene, x) -> mi.TensorXf:
        with dr.profile_range("render"):
            result = mi.render(scene, spp=1)
        return result

    w = 1024
    h = 1024

    scene = mi.cornell_box()
    # del scene["large-box"]
    # del scene["small-box"]
    scene["sensor"]["film"]["width"] = w
    scene["sensor"]["film"]["height"] = h

    print(f"{scene=}")
    scene = mi.load_dict(scene)

    params = mi.traverse(scene)
    print(params)

    b = 0
    n = 3

    k = "light.emitter.radiance.value"
    value = mi.Float(params[k].x)

    t_ref = 0
    t_ref_exec = 0

    for i in range(b + n):
        params[k].x = value + 10.0 * i
        params.update()

        dr.kernel_history_clear()

        dr.sync_thread()
        start = time.time()
        reference = func(scene, params[k].x)
        dr.eval(reference)
        dr.sync_thread()
        end = time.time()

        history = dr.kernel_history()
        execution_time = sum(
            [k["execution_time"] for k in history if "execution_time" in k]
        )
        mi.util.write_bitmap(f"out/reference{i}.jpg", reference)
        print(
            f"reference total: {end - start}s, execution_time: {execution_time/1000}s"
        )
        if i >= b:
            t_ref += end - start
            t_ref_exec += execution_time

    frozen = dr.freeze(func)

    t_frozen = 0

    for i in range(b + n):
        params[k].x = value + 10.0 * i
        params.update()

        dr.sync_thread()
        start = time.time()
        result = frozen(scene, (params[k].x))
        dr.eval(result)
        dr.sync_thread()
        end = time.time()

        mi.util.write_bitmap(f"out/result{i}.jpg", result)
        print(f"frozen took: {(end - start)}s")

        if i >= b:
            t_frozen += end - start

    t_ref_exec /= 1000
    print(f"avg: {t_ref/n=}s, {t_ref_exec/n=}s, {t_frozen/n=}")
