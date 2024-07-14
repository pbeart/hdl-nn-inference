import math

import argparse
import pickle
import warnings

import torch
from mlgen import SequentialStepHDL, DenseLogLayer, ReLUStep, Model, BiasStep, DenseLayer


def identify_layer(layer):
    if isinstance(layer, torch.jit.ScriptModule):
        return layer.original_name
    else:
        if isinstance(layer, torch.nn.Linear):
            return "Linear"
        else:
            raise NotImplementedError(f"Don't know how to deal with {layer}")
        

def torch_model_to_mlgen(model, log_quantize = False, log_quantize_precision = None) -> Model:
    layers = []
    # todo: don't always make auto-relu layers!
    auto_relu = True

    if auto_relu:
        warnings.warn("Automatically creating ReLU activation layers because no known activation type.")

    input_count = None
    output_count = None
    
    first_iteration = True

    for index, layer in enumerate(model.children()):
        print(f"Processing layer #{index}...")
        name = identify_layer(layer)
        if name == "Linear":
            if first_iteration:
                input_count = len(layer.state_dict()["weight"][0])
                first_iteration = False

            output_count = len(layer.state_dict()["weight"])

            if log_quantize:
                mlgen_layer = make_log_mult_layer(layer.state_dict()["weight"], log_quantize_precision)
                
                layers.append(mlgen_layer)
            else:
                mlgen_layer = DenseLayer(layer.state_dict()["weight"])
                
                layers.append(mlgen_layer)
            
            layers.append(BiasStep(layer.state_dict()["bias"]))
        else:
            raise NotImplementedError(f"Don't know how to convert {layer}")
        if auto_relu:
            layers.append(ReLUStep())

    return Model(layers, input_count, output_count)

def make_log_mult_layer(weights, precision):
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
                power = math.ceil(math.log2(abs(target_weight)))
                negate = target_weight < 0
                this_weight = math.pow(2, power)
                target_weight -= (-1 if negate else 1) * this_weight

                this_log_weights_signs.append(
                    [power, negate]
                )

            neuron_weights_signs.append(this_log_weights_signs)

        log_weights_signs.append(neuron_weights_signs)

    print("Produced denseloglayer of size ", len(log_weights_signs), len(log_weights_signs[0]))
    return DenseLogLayer(log_weights_signs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some images')
    parser.add_argument("model", help="The input torchfile")
    parser.add_argument("destination", help="Destination .mlgen file")

    parser.add_argument("--log-quantize", help="", action="store_true")
    parser.add_argument("--log-quantize-precision", type=float)

    args = parser.parse_args()

    if args.log_quantize:
        if args.log_quantize_precision is None:
            raise ValueError("Must provide --log-quantize-precision if --log-quantize is used!")

    model = torch.jit.load(args.model, map_location='cpu')
    mlgen_model = torch_model_to_mlgen(model, args.log_quantize, args.log_quantize_precision)

    with open(args.destination, "wb") as f:
        pickle.dump(mlgen_model, f)