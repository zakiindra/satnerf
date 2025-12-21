import argparse
import json

import tensorrt as trt


def main(args):
    trt_logger = trt.Logger(trt.Logger.ERROR)

    # Load the engine
    with open(args.engine, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    # Create an inspector
    inspector = engine.create_engine_inspector()

    layer_json_str = inspector.get_engine_information(
        trt.LayerInformationFormat.JSON
    )
    
    print(layer_json_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a TensorRT engine")
    parser.add_argument("--engine", help="Path to the engine file", required=True)
    args = parser.parse_args()
    main(args)
