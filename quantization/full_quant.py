import argparse
from pathlib import Path

import tensorrt as trt

from quantization.calibrator import RandomEntropyCalibrator


def export_to_trt(onnx_path, precision, output_path):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()

    # === Parse ONNX ===
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed.")

    # === Inspect layers (optional per-layer precision control) ===
    # for i in range(network.num_layers):
    #     layer = network.get_layer(i)
    #     chosen_precision = trt.DataType.HALF if precision == "fp16" else trt.DataType.INT8
    #     layer.precision = chosen_precision
    #     for j in range(layer.num_outputs):
    #         layer.set_output_type(j, chosen_precision)

    # === Enable Precision Modes ===
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        calibration_cache = './generated/calib.cache'
        calib = RandomEntropyCalibrator(cache_file=calibration_cache, seed=42)
        config.int8_calibrator = calib
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS);

    # === Define Optimization Profile ===
    profile = builder.create_optimization_profile()
    input_shapes = {
        "input_xyz":   ([1, 3], [1024, 3], [1310720, 3]),
        "input_sun_dir": ([1, 3], [1024, 3], [1310720, 3]),
        "input_t":     ([1, 4], [1024, 4], [1310720, 4]),
    }

    for name, (min_shape, opt_shape, max_shape) in input_shapes.items():
        profile.set_shape(name, min=min_shape, opt=opt_shape, max=max_shape)

    config.add_optimization_profile(profile)

    # === Build the engine ===
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed")

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)

    # === Save engine to file ===
    # BASE_DIR = Path(__file__).resolve().parents[1]
    # out_dir = BASE_DIR / "generated/models"
    # out_dir.mkdir(parents=True, exist_ok=True)

    # out_path = out_dir / f"model_{precision}.trt"
    with open(output_path, "wb") as f:
        f.write(engine.serialize())

    print("TensorRT engine built successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Path to ONNX model")
    parser.add_argument("--precision", help="Precision to target")
    parser.add_argument("--output", "-o", help="Path for output .trt engine")

    args = parser.parse_args()

    main(args)
