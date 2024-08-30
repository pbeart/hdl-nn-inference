import math

import argparse
import pickle
import warnings

from mlgen import DenseLogLayer, ReLUStep, Model, BiasStep, DenseLayer, WeightIncrementalLogLayer, WeightFragment

from litob import litob_converter

def identify_layer(layer):
    if isinstance(layer, torch.jit.ScriptModule):
        return layer.original_name
    else:
        if isinstance(layer, torch.nn.Linear):
            return "Linear"
        else:
            raise NotImplementedError(f"Don't know how to deal with {layer}")
        
def convert_layer(layer, log_quantize=False, log_incremental = False, log_quantize_precision: float = 0.1, one_bit=False):

    if len([opt for opt in [log_quantize, log_incremental, one_bit] if opt]) > 1:
        raise ValueError("Can only specify one option flag!")
    
    name = identify_layer(layer)
    layers = []
    if name == "Linear":
        input_count = len(layer.state_dict()["weight"][0])

        if log_incremental:
            print("Making incremental log layer...")
            layers.append(WeightIncrementalLogLayer(make_log_mult_layer(layer.state_dict()["weight"], log_quantize_precision)))
        else:
            if log_quantize:
                print("Making log-quantized layer")
                layers.append(DenseLogLayer(make_log_mult_layer(layer.state_dict()["weight"], log_quantize_precision)))
            else:
                print("Making FullyConnected layer")
                layers.append(DenseLayer(layer.state_dict()["weight"]))
        
        layers.append(BiasStep(layer.state_dict()["bias"]))
        
        return layers
    else:
        raise NotImplementedError(f"Don't know how to convert {layer}")


def make_log_mult_layer(weights: list[list[float]], precision: float) -> list[list[list[WeightFragment]]]:
    print("Got weights of size", len(weights), "x", len(weights[0]))
    assert precision > 0

    log_weights_signs = []

    for neuron_weights in weights:
        neuron_weights_signs = []
        for weight in neuron_weights:
            this_log_weights_signs = []

            # Weight which must be added to achieve desired multiplication
            target_weight = weight

            while abs(target_weight) > precision:
                exponent = math.ceil(math.log2(abs(target_weight)))
                negate = target_weight < 0
                this_weight_contribution = math.pow(2, exponent)
                target_weight -= (-1 if negate else 1) * this_weight_contribution

                this_log_weights_signs.append(
                    WeightFragment(exponent=exponent, negative=negate)
                )

            neuron_weights_signs.append(this_log_weights_signs)

        log_weights_signs.append(neuron_weights_signs)

    #print("Produced denseloglayer of size ", len(log_weights_signs), len(log_weights_signs[0]))
    return log_weights_signs

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

        layers += convert_layer(layer, log_incremental=layer_log_incremental)
        first_iteration = False
        if auto_relu and not last_iteration:
            layers.append(ReLUStep())

    return Model(layers, input_count, output_count)

def converter_partial(converter):
    def partial(args):
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

    subparsers = parser.add_subparsers(dest="converter")

    default_parser = subparsers.add_parser("default")

    default_parser.add_argument("--log-quantize-all", help="", action="store_true")
    default_parser.add_argument("--log-quantize-precision", "-lp", type=float)
    default_parser.add_argument("--first-layer-log-incremental", "-i", action="store_true")

    default_parser.set_defaults(func=converter_partial(default_converter))

    litob_parser = subparsers.add_parser("litob") # litob = "log-incremental then one bit", also a kind of saltwater clam
    litob_parser.set_defaults(func=converter_partial(litob_converter))

    
    
    parser.parse_args()
    

# also need to import it here, for global-scope availability for use in module functions
import torch