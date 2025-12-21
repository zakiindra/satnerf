import argparse

import onnx
import tensorrt as trt


parser = argparse.ArgumentParser()
parser.add_argument("--path", "-p")

args = parser.parse_args()

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
onnx_path = args.path

builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# Load the ONNX model
def parse_onnx(network, path, logger):
    parser = trt.OnnxParser(network, logger)
    with open(path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed.")

    return network


def get_elementwise_op(layer):
    if isinstance(layer, trt.IElementWiseLayer):
        return layer.operation
    return None


def print_layer_types(network):
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        print(f"Layer {i}: Name = {layer.name}, Type = {layer.type}")
        if layer.type == trt.LayerType.ELEMENTWISE:
            try:
                ew_layer = trt.IElementWiseLayer(layer)
                print(f"Layer {i}: {layer.name}, ElementWise Operation: {ew_layer.op}")
            except Exception as e:
                print(f"Layer {i}: {layer.name}, failed to cast: {e}")
        # if layer.type == trt.LayerType.ELEMENTWISE:
        #     op = get_elementwise_op(layer)
        #     print(f"\tLayer {i}: {layer.name} - ElementWise Op: {op}")


network = parse_onnx(network, args.path, TRT_LOGGER)
print_layer_types(network)
