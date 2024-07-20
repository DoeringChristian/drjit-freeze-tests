import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")


def f(x, y, i: mi.UInt32) -> mi.Color3f:
    return dr.gather(mi.Float, x, i % 2) * mi.Float(i) * y


@dr.syntax(print_code=True)
def loop(x: mi.Float, y: mi.Float, n: int = 10):
    y, i = mi.Float(0), mi.UInt(0)

    while dr.hint(i < n, max_iterations=-1):
        y += f(x, y, i)
        i += 1

    return y


if __name__ == "__main__":
    # dr.set_log_level(dr.LogLevel.Trace)
    # dr.set_flag(dr.JitFlag.Debug, True)
    dr.set_flag(dr.JitFlag.ReuseIndices, False)
    dr.set_flag(dr.JitFlag.LaunchBlocking, True)

    key = "value"

    x = dr.opaque(mi.Float, 1, shape=3)
    y = dr.opaque(mi.Float, 2, shape=10)

    ref = loop(x, y)
    print(f"{ref=}")

    x = dr.opaque(mi.Float, 0, shape=3)
    y = dr.opaque(mi.Float, 2, shape=10)

    opt = mi.ad.Adam(lr=0.05, params={key: x})
    dr.enable_grad(opt[key])

    def mse(res):
        return dr.mean(dr.square(res - ref))

    for it in range(50):
        x = opt[key]

        res = loop(x, y)

        loss = mse(res)

        dr.backward(loss)

        print(f"{it=}, {loss=}, {dr.grad(x)=}")

        opt.step()
