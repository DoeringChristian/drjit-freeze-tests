import mitsuba as mi
import drjit as dr

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")


@dr.syntax(print_code=True)
def loop(tex: mi.Texture, x: mi.Float):

    i = mi.UInt(0)
    y = dr.full(mi.Float, 0, 10)

    while dr.hint(i < 10, max_iterations=-1):
        y += tex.eval(mi.SurfaceInteraction3f()).x * x
        i += 1

    return y


if __name__ == "__main__":

    tex: mi.Texture = mi.load_dict(
        {
            "type": "srgb",
            "color": [1.0, 0.0, 0.0],
        }
    )

    x = dr.opaque(mi.Float, 1, 10)
    dr.enable_grad(x)

    opt = mi.ad.Adam(lr=0.01, params={"x": x})

    for i in range(10):
        x = opt["x"]

        y = loop(tex, x)

        loss = dr.sum(y)

        dr.backward(y)

        opt.step()

    print(f"{x=}")
