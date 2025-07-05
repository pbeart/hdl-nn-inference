import warnings
import conversionutils
from mlgen import FullyConnectedLogLayer, ReLUStep, Model, BiasStep, FullyConnectedLayer, WeightIncrementalLogLayer, WeightFragment

def litob_converter(model, args):
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
        name = conversionutils.identify_layer(layer)

        last_iteration = index == (len(torch_layers) - 1)

        if first_iteration:
            # Make log-incremental layer
            layers += conversionutils.convert_layer(layer) 
        else:
            # Make one-bit layer
            layers += conversionutils.convert_layer(layer, one_bit=True)

        
        first_iteration = False
        if auto_relu and not last_iteration:
            layers.append(ReLUStep())

    return Model(layers, input_count, output_count)