import mitsuba as mi
import drjit as dr
import gc

dr.set_log_level(dr.LogLevel.Trace)
mi.set_variant("cuda_ad_rgb")


@dr.syntax(print_code=True)
def main():
    x = mi.Float(1, 2)
    dr.enable_grad(x)
    print(f"{dr.grad_enabled(x)=}")
    print(f"x: a{x.index_ad} r{x.index}")

    with dr.suspend_grad():
        print(f"x: a{x.index_ad} r{x.index}")
        i = mi.UInt(0)

        while dr.hint(i < 3, mode="symbolic"):
            x += 1
            i += 1


if __name__ == "__main__":
    main()
