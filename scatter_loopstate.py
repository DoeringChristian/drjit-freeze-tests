import drjit as dr
import mitsuba as mi
from dataclasses import dataclass

mi.set_variant("cuda_ad_rgb")


# @dataclass
class Test:
    x: mi.Float

    DRJIT_STRUCT = {"x": mi.Float}

    def __init__(self) -> None:
        self.x = dr.opaque(mi.Float, 0, shape=10)


@dr.syntax(print_code=True)
def loop(x: mi.Float, n: int = 10):

    y, i = mi.Float(2), mi.UInt(0)

    while dr.hint(i < n, max_iterations=-1):
        y *= y
        dr.scatter(x.x, y, i)
        i += 1

    return x


if __name__ == "__main__":
    # dr.set_log_level(dr.LogLevel.Trace)
    # dr.set_flag(dr.JitFlag.Debug, True)
    dr.set_flag(dr.JitFlag.ReuseIndices, False)
    dr.set_flag(dr.JitFlag.LaunchBlocking, True)

    # x = dr.opaque(mi.Float, 0, shape=10)

    x = Test()

    loop(x)

    print(f"{x.x=}")
