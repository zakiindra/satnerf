import argparse
import colorama
from colorama import Fore, Style
colorama.init(autoreset=True)
import time

import tensorrt as trt
from quantization.calibrator import RandomEntropyCalibrator


def parse_onnx(network, path, logger):
    parser = trt.OnnxParser(network, logger)
    with open(path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed.")

    return network


def set_tested_precisions(precision: str):
    precision_map = {
        "fp16": trt.DataType.HALF,
        "int8": trt.DataType.INT8,
    }

    tested_precision = precision_map[precision]
    return precision, tested_precision


def set_layers_precision(network, tested_precision, layer_group_names: str):
    layer_groups = {
        "group1": [
            "/fc_net.0/Gemm",
            "/fc_net.1/Mul",
            "/fc_net.1/Sin",
            "/fc_net.2/Gemm",
            "/fc_net.3/Mul",
            "/fc_net.3/Sin",
            "/fc_net.4/Gemm",
            "/fc_net.3_1/Mul",
            "/fc_net.3_1/Sin",
            "/fc_net.6/Gemm",
            "/fc_net.3_2/Mul",
            "/fc_net.3_2/Sin",
        ],
        "group2": [
            "/fc_net.8/Gemm",
            "/fc_net.3_3/Mul",
            "/fc_net.3_3/Sin",
            "/fc_net.10/Gemm",
            "/fc_net.3_4/Mul",
            "/fc_net.12/Gemm",
            "/fc_net.3_5/Mul",
            "/fc_net.3_5/Sin",
            "/fc_net.14/Gemm",
            "/fc_net.3_6/Mul",
            "/fc_net.3_6/Sin",
        ],
        "group3": ["/feats_from_xyz/Gemm"],
        "group4": [
            "/sun_v_net/sun_v_net.0/Gemm",
            "/sun_v_net/sun_v_net.1/Mul",
            "/sun_v_net/sun_v_net.1/Sin",
            "/sun_v_net/sun_v_net.2/Gemm",
            "/sun_v_net/fc_net.3/Mul",
            "/sun_v_net/fc_net.3/Sin",
            "/sun_v_net/sun_v_net.4/Gemm",
            "/sun_v_net/fc_net.3_1/Mul",
            "/sun_v_net/fc_net.3_1/Sin",
            "/sun_v_net/sun_v_net.6/Gemm",
            "/sun_v_net/sun_v_net.7/Sigmoid"
        ],
        "group5": [
            "/sigma_from_xyz/sigma_from_xyz.0/Gemm",
            "/sigma_from_xyz/sigma_from_xyz.1/Softplus",
        ],
        "group6": [
            "/rgb_from_xyzdir/rgb_from_xyzdir.0/Gemm",
            "/rgb_from_xyzdir/fc_net.3/Mul",
            "/rgb_from_xyzdir/fc_net.3/Sin",
            "/rgb_from_xyzdir/rgb_from_xyzdir.2/Gemm",
            "/rgb_from_xyzdir/rgb_from_xyzdir.3/Sigmoid",
            "/Mul",
            "/Sub"
        ],
        "group7": [
            "/beta_from_xyz/beta_from_xyz.0/Gemm",
            "/beta_from_xyz/fc_net.3/Mul",
            "/beta_from_xyz/fc_net.3/Sin",
            "/beta_from_xyz/beta_from_xyz.2/Gemm",
            "/beta_from_xyz/beta_from_xyz.3/Softplus",
        ],
        "group8": [
            "/sky_color/sky_color.0/Gemm",
            "/sky_color/sky_color.1/Relu",
            "/sky_color/sky_color.2/Gemm",
            "/sky_color/sky_color.3/Sigmoid"
        ]
    }

    groups_to_quantize = []
    for lgroup in layer_group_names.split(sep=","):
        groups_to_quantize.extend(layer_groups[lgroup])

    # print("-" * 80)
    # print("\nONNX Network Information\n")
    # print("-" * 80)
    # for i in range(network.num_layers):
    #     layer = network.get_layer(i)
    #     print(f"{i}: {layer.name} - {layer.type}")
    #
    #
    # print("-" * 80)
    # print("\nApplying quantization to:\n")
    # print("-" * 80)
    # layer_names = { network.get_layer(i).name for i in range(network.num_layers) }
    # for name in groups_to_quantize:
    #     if name in layer_names:
    #         print(f"{name}\t{Fore.GREEN} found")
    #     else:
    #         print(f"{name}\t{Fore.RED} not found")

    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.name in groups_to_quantize:
            layer.precision = tested_precision
            for j in range(layer.num_outputs):
                layer.set_output_type(j, tested_precision)
        else:
            layer.precision = trt.DataType.FLOAT 
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.DataType.FLOAT)

    return network


def set_builder_config(config, builder, precision_choices):
    def add_engine_profiles(builder):
        profile = builder.create_optimization_profile()

        input_shapes = {
            "input_xyz":   ([1, 3], [1310720, 3], [1310720, 3]),
            "input_sun_dir": ([1, 3], [1310720, 3], [1310720, 3]),
            "input_t":     ([1, 4], [1310720, 4], [1310720, 4]),
        }

        for name, (min_shape, opt_shape, max_shape) in input_shapes.items():
            profile.set_shape(name, min=min_shape, opt=opt_shape, max=max_shape)

        return profile

    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if precision_choices == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision_choices == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        calibration_cache = './generated/calib.cache'
        calib = RandomEntropyCalibrator(cache_file=calibration_cache, seed=42)
        config.int8_calibrator = calib

    optimization_profiles = add_engine_profiles(builder)
    config.add_optimization_profile(optimization_profiles)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS);

    return config


def build_engine(builder, network, config, logger):
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)

    # === Save engine to file ===
    with open(args.output, "wb") as f:
        f.write(engine.serialize())


def main(args):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    onnx_path = args.path

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # Configure precisions to use
    precision_choice, tested_precision = set_tested_precisions(args.precision)

    # Set up builder config
    config = builder.create_builder_config()
    config = set_builder_config(config, builder, precision_choice)

    start = time.time()

    # Parse ONNX
    network = parse_onnx(network, onnx_path, TRT_LOGGER)

    # Set per-layer precision
    network = set_layers_precision(network, tested_precision, args.layer_group_name)

    # Build the engine
    build_engine(builder, network, config, TRT_LOGGER)
    print(f"{Fore.CYAN}Quantization complete")

    delta = time.time() - start
    print(f"Conversion took {delta}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p")
    parser.add_argument("--output", "-o")
    parser.add_argument("--precision", type=str, default="fp16",
                        help="Precision mode to try: fp16,int8")
    parser.add_argument("--layer-group-name", "-lgn",
        help="Comma-separated layer groups to quantize i.e. group_1,group_2")

    args = parser.parse_args()

    main(args)

    print("TensorRT engine built successfully!")
