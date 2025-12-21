import argparse
import random
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


def set_available_precisions(args):
    precision_map = {
        "fp32": trt.DataType.FLOAT,
        "fp16": trt.DataType.HALF,
        "int8": trt.DataType.INT8,
    }

    precision_cls = precision_map[args.precision]
    return args.precision, precision_cls 


def set_layers_precision(network, precision_cls):
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        layer.precision = precision_cls
        for j in range(layer.num_outputs):
            layer.set_output_type(j, precision_cls)

        # gemms = {
        #     trt.LayerType.MATRIX_MULTIPLY,
        #     trt.LayerType.SCALE,
        #     trt.LayerType.ELEMENTWISE
        # }

        # if layer.type in gemms:
        #     chosen_precision = random.choice(available_precisions)
        #     layer.precision = chosen_precision
        #     for j in range(layer.num_outputs):
        #         layer.set_output_type(j, chosen_precision)

    return network


def set_builder_config(config, builder, precision_str):
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
    if precision_str == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    if precision_str == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        calibration_cache = './generated/calib.cache'
        calib = RandomEntropyCalibrator(cache_file=calibration_cache, seed=42)
        config.int8_calibrator = calib

    optimization_profiles = add_engine_profiles(builder)
    config.add_optimization_profile(optimization_profiles)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS);

    return config


def build_engine(builder, network, config, logger):
    print("1")
    serialized_engine = builder.build_serialized_network(network, config)
    print("2")
    if serialized_engine is None:
        raise RuntimeError("Engine build failed")

    runtime = trt.Runtime(logger)
    print("3")
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    print("4")

    # === Save engine to file ===
    print("5")
    with open(args.output, "wb") as f:
        f.write(engine.serialize())
    print("6")


def main(args):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    onnx_path = args.path

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # Configure precisions to use
    precision_str, precision_cls = set_available_precisions(args)

    # Set up builder config
    config = builder.create_builder_config()
    config = set_builder_config(config, builder, precision_str)

    start = time.time()

    # Parse ONNX
    network = parse_onnx(network, onnx_path, TRT_LOGGER)

    # Set per-layer precision
    network = set_layers_precision(network, precision_cls)

    # Build the engine
    build_engine(builder, network, config, TRT_LOGGER)

    delta = time.time() - start
    print(f"Conversion took {delta}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p")
    parser.add_argument("--output", "-o")
    parser.add_argument("--precision", type=str, default="fp16")

    args = parser.parse_args()

    main(args)

    print("TensorRT engine built successfully!")
