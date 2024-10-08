import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")

class MySampler(mi.Sampler):
    def __init__(self, props) -> None:
        super().__init__(props)
        self.value = mi.Float(0.5)
        dr.make_opaque(self.value)

    def next_1d(self, active: mi.Bool = True):
        return self.value


mi.register_sampler("mysampler", lambda props: MySampler(props))

if __name__ == "__main__":
    sampler = 
    ...
