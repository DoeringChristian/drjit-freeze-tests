import mitsuba as mi
import drjit as dr
import time

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

    independent = mi.load_dict({"type": "independent"})
    print(f"{type(independent)=}")
    # print(f"{independent.__dict__=}")

    print(f"{dir(mi.Sampler)=}")

    def func(sampler: MySampler) -> mi.Float:
        return sampler.next_1d()

    frozen = dr.freeze(func)

    sampler: mi.Sampler = mi.load_dict({"type": "mysampler"})
    print(f"{sampler.__dict__=}")

    print(f"{frozen(sampler)}")
