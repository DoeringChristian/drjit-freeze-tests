import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")


def f(texture: mi.Texture, i: mi.UInt32) -> mi.Color3f:
    si = mi.SurfaceInteraction3f()
    return texture.eval(si) + mi.Float(i)


@dr.syntax
def spec_avg(texture: mi.Texture) -> mi.Color3f:
    img = mi.Color3f(0)
    i = mi.UInt32(0)

    while dr.hint(i < 10, max_iterations=-1):
        img += f(texture, i)
        i += 1

    img /= 10

    return img


if __name__ == "__main__":
    # dr.set_log_level(dr.LogLevel.Trace)
    # dr.set_flag(dr.JitFlag.Debug, True)
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

    ref = spec_avg(srgb)

    params[key] = mi.Color3f(0.0, 1.0, 0.0)
    params.update()

    opt = mi.ad.Adam(lr=0.05)
    opt[key] = params[key]
    params.update(opt)

    def mse(image):
        return dr.mean(dr.mean(dr.square(image - ref)))

    for it in range(50):
        img = spec_avg(srgb)

        loss = mse(img)
        print(f"{it=}, {loss=}")

        dr.backward(loss)

        opt.step()

        opt[key] = dr.clamp(opt[key], 0.0, 1.0)

        params.update(opt)
