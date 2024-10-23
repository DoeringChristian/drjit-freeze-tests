import mitsuba as mi
import drjit as dr
import os

mi.set_variant("cuda_ad_rgb")

if __name__ == "__main__":
    w = 1024
    h = 1024

    scene = mi.load_file("scenes/staircase/scene_v3.xml", resx=w, resy=h)

    def func(scene: mi.Scene, seed: mi.UInt32):
        return mi.render(scene, spp=1, seed=seed)

    frozen = dr.freeze(func)

    n = 10

    for i in range(n):
        img = frozen(scene, mi.UInt32(i))

        os.makedirs("out/seed/", exist_ok=True)
        mi.util.write_bitmap(f"out/seed/{i}.exr", img)

    print(f"{frozen.n_recordings=}")
