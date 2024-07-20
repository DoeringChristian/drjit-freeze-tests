import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")


def f(x, tex, i: mi.UInt32) -> mi.Color3f:
    si = mi.SurfaceInteraction3f()
    return x * mi.Float(i) / 2 + tex.eval(si).x


@dr.syntax(print_code=True)
def loop(x: mi.Float, tex, n: int = 10):
    y, i = mi.Float(0), mi.UInt(0)

    while dr.hint(i < n, max_iterations=-1):
        y += f(x, tex, i)
        i += 1

    return y


if __name__ == "__main__":
    # dr.set_log_level(dr.LogLevel.Trace)
    # dr.set_flag(dr.JitFlag.Debug, True)
    dr.set_flag(dr.JitFlag.ReuseIndices, False)
    dr.set_flag(dr.JitFlag.LaunchBlocking, True)

    srgb: mi.Texture = mi.load_dict(
        {
            "type": "srgb",
            "color": [1.0, 0.0, 0.0],
        }
    )
    print(f"{type(srgb)=}")

    key = "value"
    n = 10

    x = dr.opaque(mi.Float, 1, shape=n)

    ref = loop(x, srgb)
    print(f"{ref=}")

    x = dr.opaque(mi.Float, 0, shape=n)

    opt = mi.ad.Adam(lr=0.05, params={key: x})
    dr.enable_grad(opt[key])

    def mse(res):
        return dr.mean(dr.square(res - ref))

    for it in range(50):
        x = opt[key]

        res = loop(x, srgb)

        loss = mse(res)

        dr.backward(loss)

        print(f"{it=}, {loss=}, {dr.grad(x)=}")

        opt.step()
