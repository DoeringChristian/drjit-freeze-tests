import mitsuba as mi
import drjit as dr
import time
import numpy as np

mi.set_variant("cuda_ad_rgb")
# mi.set_variant("llvm_ad_rgb")

# import mypath

if __name__ == "__main__":
    # dr.set_log_level(dr.LogLevel.Trace)
    # dr.set_log_level(dr.LogLevel.Debug)
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    dr.set_flag(dr.JitFlag.VCallOptimize, True)
    dr.set_flag(dr.JitFlag.SymbolicCalls, True)
    dr.set_flag(dr.JitFlag.Debug, True)
    dr.set_flag(dr.JitFlag.ReuseIndices, False)
    dr.set_flag(dr.JitFlag.LaunchBlocking, True)

    def func(scene: mi.Scene, x) -> mi.TensorXf:
        with dr.profile_range("render"):
            result = mi.render(scene, spp=1)
        return result

    w = 128
    h = 128

    scene = {
        "type": "scene",
        "integrator": {
            "type": "path",
            "max_depth": 12,
        },
        "sensor": {
            "type": "perspective",
            "fov_axis": "x",
            "fov": 39.6,
            "principal_point_offset_x": 0.0,
            "principal_point_offset_y": 0.0,
            "near_clip": 0.1,
            "far_clip": 100.0,
            "to_world": mi.ScalarTransform4f()
            .rotate([1.0, 0.0, 0.0], -153.56)
            .rotate([0.0, 1.0, 0.0], -46.69)
            .rotate([0.0, 0.0, 1.0], -179.99)
            .translate([7.35, 4.95, 6.93]),
            "sampler": {
                "type": "independent",
                "sample_count": 1,
            },
            "film": {"type": "hdrfilm", "width": w, "height": h},
        },
        "material": {
            "type": "diffuse",
            "reflectance": {"type": "rgb", "value": [0.8, 0.8, 0.8]},
        },
        "emitter": {
            "type": "rectangle",
            "emitter": {
                "type": "area",
                "radiance": {
                    "type": "rgb",
                    "value": [250.0, 250.0, 250.0],
                },
            },
        },
        # "box": {
        #     "type": "ply",
        #     "filename": "scenes/cube/meshes/Cube.ply",
        #     "face_normals": True,
        #     "bsdf": {
        #         "type": "ref",
        #         "id": "material",
        #     },
        # },
        "box": {"type": "cube", "bsdf": {"type": "ref", "id": "material"}},
    }

    # mi.xml.dict_to_xml(mi.cornell_box(), "scenes/cube/cbox.xml")
    print(f"{mi.cornell_box()=}")

    # scene = mi.load_file("scenes/cube/scene-py.xml")
    # scene = mi.load_dict(scene)

    scene = mi.load_file("scenes/cube/scene.xml", resx=w, resy=h)

    params = mi.traverse(scene)
    print(params)
    # exit()

    # print(f"{scene.shapes_dr()=}")
    # exit()
    b = 0
    n = 3

    k = "elm__4.emitter.radiance.value"
    # k = "emitter.emitter.radiance.value"
    value = mi.Float(params[k].x)

    t_ref = 0
    t_ref_exec = 0

    for i in range(b + n):
        params[k].x = value + 10.0 * i
        params.update()

        dr.kernel_history_clear()

        dr.sync_thread()
        start = time.time()
        reference = func(scene, params[k].x)
        dr.eval(reference)
        dr.sync_thread()
        end = time.time()

        history = dr.kernel_history()
        execution_time = sum(
            [k["execution_time"] for k in history if "execution_time" in k]
        )
        mi.util.write_bitmap(f"out/reference{i}.jpg", reference)
        print(
            f"reference total: {end - start}s, execution_time: {execution_time/1000}s"
        )
        if i >= b:
            t_ref += end - start
            t_ref_exec += execution_time

    frozen = dr.freeze(func)

    t_frozen = 0

    for i in range(b + n):
        params[k].x = value + 10.0 * i
        params.update()

        dr.sync_thread()
        start = time.time()
        result = frozen(scene, (params[k].x))
        dr.eval(result)
        dr.sync_thread()
        end = time.time()

        mi.util.write_bitmap(f"out/result{i}.jpg", result)
        print(f"frozen took: {(end - start)}s")

        if i >= b:
            t_frozen += end - start

    t_ref_exec /= 1000
    print(f"avg: {t_ref/n=}s, {t_ref_exec/n=}s, {t_frozen/n=}")
