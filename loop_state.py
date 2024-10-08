import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")


@dr.syntax(print_code=True)
def loop(x: list[mi.Float], y: mi.Float, n: int = 10) -> mi.Float:

    i = mi.UInt(0)
    while dr.hint(i < n):
        y += dr.gather(mi.Float, x[0], mi.UInt(y))
        i += 1

    return y


if __name__ == "__main__":
    dr.set_log_level(dr.LogLevel.Trace)
    dr.set_flag(dr.JitFlag.Debug, True)
    dr.set_flag(dr.JitFlag.ReuseIndices, False)
    dr.set_flag(dr.JitFlag.LaunchBlocking, True)
    # dr.set_flag(dr.JitFlag.LoopOptimize, True)

    x = dr.arange(mi.Float, 10)
    y = dr.arange(mi.Float, 5)
    print(f"{x.index=}")
    print(f"{y.index=}")

    dr.make_opaque(x, y)

    y = loop([x, x], y)
