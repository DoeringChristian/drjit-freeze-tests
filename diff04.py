import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")


def f(texture: mi.Texture, i: mi.UInt32) -> mi.Color3f:
    si = mi.SurfaceInteraction3f()
    return texture.eval(si) + mi.Float(i)


@dr.syntax(print_code=True)
def spec_avg(texture: mi.Texture) -> mi.Color3f:
    res = mi.Color3f(0)
    # i = mi.UInt32(0)
    i = dr.arange(mi.UInt, 10)
    dr.make_opaque(i)

    while dr.hint(i < 10, max_iterations=-1):
        res += f(texture, i)
        i += 1

    res /= 10

    return res


if __name__ == "__main__":
    # dr.set_log_level(dr.LogLevel.Trace)
    # dr.set_flag(dr.JitFlag.Debug, True)
    dr.set_flag(dr.JitFlag.LoopOptimize, True)
    dr.set_flag(dr.JitFlag.ReuseIndices, False)
    dr.set_flag(dr.JitFlag.LaunchBlocking, True)

    use_loop = True

    srgb: mi.Texture = mi.load_dict(
        {
            "type": "srgb",
            "color": [1.0, 0.0, 0.0],
        }
    )

    print(f"{type(srgb)=}")

    dr.schedule(srgb)

    params = mi.traverse(srgb)

    print(f"{params=}")

    key = "value"

    # spec_avg = dr.freeze(spec_avg)

    ref = spec_avg(srgb)
    print(f"{ref=}")

    params[key] = mi.Color3f(0.0, 1.0, 0.0)
    dr.make_opaque(params[key])
    params.update()

    opt = mi.ad.Adam(lr=0.05)
    opt[key] = params[key]
    params.update(opt)

    def mse(image):
        return dr.mean(dr.mean(dr.square(image - ref)))

    for it in range(50):
        res = spec_avg(srgb)

        loss = mse(res)
        print(f"{it=}, {loss=}, {res=}")

        dr.backward(loss)

        opt.step()

        opt[key] = dr.clip(opt[key], 0.0, 1.0)

        params.update(opt)
