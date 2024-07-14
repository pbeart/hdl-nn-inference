import pickle
import mlgen
import fp as fp
import hdlgen
import argparse, os

from dataclasses import dataclass

parser = argparse.ArgumentParser(description='Process some images')
parser.add_argument("model", help="the input .mlgen array")
parser.add_argument("destination", help="the destination .sv file")

# import after for better responsiveness
import numpy as np
import torch

float_environment = fp.FloatEnvironment("binary16")

args = parser.parse_args()

f = open(args.model, "rb")
mlgen_model: mlgen.Model = pickle.load(f)
f.close()

destination_filename = os.path.basename(args.destination)

if not destination_filename.endswith(".sv"):
    raise ValueError("Destination must be a .sv file, to determine module name")

module = hdlgen.Module(destination_filename.split(".sv")[0])

is_first = True

prev_layer = []

input_layer = []
output_layer = []

for layer_index, layer in enumerate(mlgen_model.layers):
    if is_first:
        input_layer = [module.AddWire(float_environment.float_size, f"network_in_{wire}") for wire in range(mlgen_model.input_count)]
        prev_layer = input_layer

        is_first = False

    is_last = layer_index == len(mlgen_model.layers) - 1

    print("Passing in size of", len(prev_layer))
    prev_layer = layer.apply(prev_layer, module, float_environment)
    print("Got layer of size", len(prev_layer), "on iteration #", layer_index)

    if is_last:
        output_layer = prev_layer
        
# set up packed arrays for input and output
input_array = module.AddInput("input_array", 16, len(input_layer))
for index, input_neuron in enumerate(input_layer):
    module.AddAssignment(input_neuron, hdlgen.Indexing(input_array, index))
    
output_array = module.AddOutput("output_array", 16, len(output_layer))
module.AddAssignment(output_array, hdlgen.Concatenation(output_layer))

#__import__("code").interact(local=locals())


with open(args.destination, "w") as sv_output:
    sv_output.write(module.hdl())

try:
    module.checks()
except Exception as e:
    raise ValueError(f"Invalid configuration, still writing to {args.destination} so you can see what's going on.")