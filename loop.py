import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")


@dr.syntax(print_code=True)
def loop(x: mi.Float, y: mi.Float, n: int = 10) -> mi.Float:

    i = mi.UInt(0)
    while dr.hint(i < n):
        y += dr.gather(mi.Float, x, 0)
        dr.scatter(x, y, i % 3)
        i += 1

    return y


if __name__ == "__main__":
    # dr.set_log_level(dr.LogLevel.Trace)
    # dr.set_flag(dr.JitFlag.Debug, True)
    dr.set_flag(dr.JitFlag.ReuseIndices, False)
    dr.set_flag(dr.JitFlag.LaunchBlocking, True)
    dr.set_flag(dr.JitFlag.LoopOptimize, True)

    key = "value"

    x = dr.arange(mi.Float, 3)
    dr.make_opaque(x)

    y = dr.arange(mi.Float, 10)
    dr.make_opaque(y)

    y = loop(x, y)

    print(y)
