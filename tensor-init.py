import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")

if __name__ == "__main__":
    dr.set_log_level(dr.LogLevel.Debug)
    print("a")
    a = mi.TensorXf(3)
    print(f"{a.array.state=}")
    print("b")
    b = mi.TensorXf(mi.Float(3))
