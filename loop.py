import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")


# def f(x, i: mi.UInt32) -> mi.Float:
#     return dr.gather(mi.Float, x, i % 2)
def f(x, i: mi.UInt32) -> mi.Float:
    return mi.Float(i) * 5


# @dr.syntax(print_code=True)
def loop(x: mi.Float, y: mi.Float, i: mi.UInt, n: int = 10) -> mi.Float:

    # while dr.hint(i < n, max_iterations=-1):
    #     y += f(x, i)
    #     dr.scatter(x, y, i % 3)
    #     # x += 1
    #     i += 1

    def _loop_cond(i, n, y):
        return i < n

    def _loop_body(i, n, y):
        y += f(x, i)
        dr.scatter(x, y, i % 3)
        i += 1
        return (i, n, y)

    "----------- invoke dr.while_loop --------------"
    from drjit import while_loop as _loop

    i, n, y = _loop(
        (i, n, y),
        _loop_cond,
        _loop_body,
        labels=("i", "n", "y"),
        max_iterations=-1,
    )
    del _loop, _loop_cond, _loop_body
    "-----------------------------------------------"

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
    dr.enable_grad(y)

    i = dr.full(mi.UInt, 0, dr.width(y))
    dr.make_opaque(i)

    y = loop(x, y, i)

    print(y)

    dr.backward(y)

    print(f"{dr.grad(y)=}")
