import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")


@dr.syntax(print_code=True)
def loop(texture: mi.Texture) -> mi.Color3f:
    res = mi.Color3f(0)
    # i = mi.UInt32(0)
    i = dr.arange(mi.UInt, 10)
    dr.make_opaque(i)

    while dr.hint(i < 10, max_iterations=-1):
        res += texture.eval(mi.SurfaceInteraction3f()) + mi.Float(i)
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

    tex: mi.Texture = mi.load_dict(
        {
            "type": "srgb",
            "color": [1.0, 0.0, 0.0],
        }
    )

    dr.schedule(tex)

    params = mi.traverse(tex)

    # spec_avg = dr.freeze(spec_avg)

    opt = mi.ad.Adam(lr=0.05)
    opt["value"] = params["value"]
    params.update(opt)

    def mse(image):
        return dr.mean(dr.mean(dr.square(image)))

    for it in range(50):
        res = loop(tex)

        loss = mse(res)

        dr.backward(loss)

        opt.step()

        params.update(opt)
