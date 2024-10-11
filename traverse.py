import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")

if __name__ == "__main__":
    dr.set_log_level(dr.LogLevel.Trace)

    def func(obj: mi.Object, *args): ...

    frozen = dr.freeze(func)

    scene = mi.cornell_box()
    # del scene["floor"]
    # del scene["ceiling"]
    # del scene["back"]
    # del scene["green-wall"]
    # del scene["red-wall"]
    # del scene["integrator"]
    # del scene["sensor"]
    # del scene["small-box"]
    # del scene["large-box"]
    # del scene["light"]["emitter"]

    print(f"{scene['light']['emitter']=}")

    # scene = {
    #     "type": "scene",
    #     "emitter": {
    #         "type": "rectangle",
    #         "emitter": {
    #             "type": "area",
    #             "radiance": {"type": "rgb", "value": [18, 14, 7]},
    #         },
    #     },
    # }

    scene = {
        "type": "rectangle",
        "emitter": {
            "type": "area",
            "radiance": {"type": "rgb", "value": [18, 14, 7]},
        },
    }

    scene = mi.load_dict(scene)

    # print(f"{scene=}")

    frozen(scene, mi.Float(1))
    frozen(scene, mi.Float(1))
