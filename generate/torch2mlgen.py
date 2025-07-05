import math

import argparse
import pickle
import warnings

from mlgen import FullyConnectedLogLayer, ReLUStep, Model, BiasStep, FullyConnectedLayer, WeightIncrementalLogLayer, WeightFragment
from conversionutils import identify_layer, convert_layer
from litob import litob_converter

def default_converter(model, args):
    if args.log_quantize_all or args.first_layer_log_incremental:
        if args.log_quantize_precision is None:
            raise ValueError("Must provide --log-quantize-precision if --log-quantize-all or --first-layer-log-incremental are used!")
        
    layers = []
    # todo: don't always make auto-relu layers!
    auto_relu = True

    if auto_relu:
        warnings.warn("Automatically creating ReLU activation layers (apart from for final layer) because no known activation type.")

    input_count = -1
    output_count = -1
    
    first_iteration = True

    torch_layers = list(model.children())

    for index, layer in enumerate(torch_layers):
        name = identify_layer(layer)

        last_iteration = index == (len(torch_layers) - 1)

        print(f"Processing layer #{index}...")
        print(repr(layer))
        print(f"Has size {len(layer.state_dict()['weight'])}x{len(layer.state_dict()['weight'][0])}")
        
        if first_iteration and args.first_layer_log_incremental:
            layer_log_incremental = True
        else:
            layer_log_incremental = False

        layers += convert_layer(layer, weight_log_incremental=layer_log_incremental)
        first_iteration = False
        if auto_relu and not last_iteration:
            layers.append(ReLUStep())

    return Model(layers, input_count, output_count)

def converter_partial(converter):
    def partial(args):
        print("parrial")
        # do this late to avoid ridiculous time to show --help
        import torch
        model = torch.jit.load(args.model, map_location='cpu')

        mlgen_model = converter(model, args)

        with open(args.destination, "wb") as f:
            pickle.dump(mlgen_model, f)
    
    return partial

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some images')
    parser.add_argument("model", help="The input torchfile")
    parser.add_argument("destination", help="Destination .mlgen file")

    subparsers = parser.add_subparsers()

    default_parser = subparsers.add_parser("default")

    default_parser.add_argument("--log-quantize-all", help="", action="store_true")
    default_parser.add_argument("--log-quantize-precision", "-lp", type=float)
    default_parser.add_argument("--first-layer-log-incremental", "-i", action="store_true")

    default_parser.set_defaults(func=converter_partial(default_converter))

    litob_parser = subparsers.add_parser("litob") # litob = "log-incremental then one bit", also a kind of saltwater clam
    litob_parser.set_defaults(func=converter_partial(litob_converter))

    
    
    args = parser.parse_args()
    args.func(args)
    

# also need to import it here, for global-scope availability for use in module functions
import torch