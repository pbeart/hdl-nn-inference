import pickle
import mlgen
import fp as fp
import argparse, os
import warnings

from dataclasses import dataclass

parser = argparse.ArgumentParser(description="Simulate the given mlgen model")
parser.add_argument("model", help="the input .mlgen array")
parser.add_argument("--incremental-log-layers", "-inc", action="store_true")
parser.add_argument("--simulate-intervals", "-int", action="store_true")
parser.add_argument("--artificial-interval", "-aint", type=float, default=0.0, help="Introduce artificial interval to network inputs of +/- this number")
#parser.add_argument("input", help="comma-separated list of inputs")


float_environment = fp.FloatEnvironment("binary16")

args = parser.parse_args()

f = open(args.model, "rb")
mlgen_model: mlgen.Model = pickle.load(f)
f.close()

#last_layer = [float(x) for x in args.input.split(",")]


num_logweight_steps = 1

def abbreviate_list(l):
    # pretty ugly
    def listify(m):
        return ", ".join(f'{float(x):.2f}' if isinstance(x,(int,float)) else str(x) for x in m)
    
    trim_to_length = 15
    if len(l) <= trim_to_length:
        return "[" + listify(l) + "]"
    else:
        each_part_length = trim_to_length//2 # e.g. 15 => two 7 length ends and a middle
        first_part = "[" + listify(l[:each_part_length])
        second_part = listify(l[-each_part_length:]) + "]"
        return first_part + " ... " + second_part + f" ({len(l)} elements)"


while True:
    last_layer = [-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.258611,1.026908,1.026908,1.026908,-0.105876,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.835989,2.490618,2.796088,2.808815,2.796088,1.128731,0.543247,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.708710,2.096053,2.643353,2.796088,2.796088,2.490618,2.312427,2.796088,1.510569,-0.093148,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.182244,0.950541,2.223332,2.796088,2.719720,2.236060,0.645071,0.174138,0.543247,2.719720,2.554257,0.046859,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,1.726943,2.796088,2.808815,2.796088,1.077820,-0.424074,-0.424074,-0.424074,-0.424074,1.103275,2.643353,1.370562,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.670527,2.808815,2.808815,2.821543,1.383290,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,1.306922,2.808815,1.803311,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.059587,2.376067,2.796088,2.796088,2.197876,-0.220427,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,1.905134,2.796088,1.192371,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.334979,0.301417,1.408745,2.796088,2.630625,1.879678,-0.105876,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.182244,2.248787,2.719720,0.683254,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.148682,2.796088,2.808815,2.681536,1.790583,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.263233,2.401522,2.236060,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,1.739671,2.796088,2.821543,2.325155,-0.029509,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,2.452434,2.796088,1.637848,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.301417,2.808815,2.808815,1.383290,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,2.032413,2.808815,2.096053,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.054964,2.134236,2.796088,2.796088,0.785078,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.296795,0.657799,2.617897,2.796088,0.479608,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,2.057869,2.796088,2.719720,1.077820,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.339601,2.796088,2.821543,1.548752,-0.258611,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.390512,2.643353,2.796088,2.439706,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.059587,2.452434,2.796088,1.854222,-0.309523,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.390512,2.643353,2.796088,1.434201,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.186866,1.688759,2.592441,1.001452,0.377784,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,1.026908,2.821543,2.490618,0.174138,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,1.828766,2.808815,2.490618,0.174138,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,1.026908,2.796088,2.312427,-0.054964,-0.424074,-0.424074,-0.054964,0.657799,0.861445,2.452434,2.808815,2.439706,0.237777,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,1.408745,2.796088,2.796088,2.325155,2.096053,2.108780,2.325155,2.796088,2.719720,2.070597,1.281466,0.046859,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.657799,2.363339,2.796088,2.796088,2.796088,2.808815,2.681536,2.439706,2.121508,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.059587,1.001452,1.001452,1.001452,1.014180,0.530519,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074]
    last_interval = [(x - args.artificial_interval, x + args.artificial_interval) for x in last_layer]

    if args.incremental_log_layers:
        print(f"Simulating with {num_logweight_steps} log-weight steps")
    reached_max_steps = False
    
    for layer_index, layer in enumerate(mlgen_model.layers):
        if isinstance(layer, mlgen.IncrementalLogLayer):
            if args.incremental_log_layers:
                reached_max_steps = reached_max_steps or layer.set_steps(num_logweight_steps)
            else:
                warnings.warn("Encountered incremental log layer but not advancing (use --incremental-log-layers)!")
        last_layer = layer.eval(last_layer)
        print(f"- Layer #{layer_index} ({layer}) got {abbreviate_list(last_layer)}")

        if args.simulate_intervals:
            last_interval = layer.eval_interval(last_interval)
            print(f"  - got intervals {abbreviate_list(last_interval)}")
    
    print(f"Output: {last_layer}")
    if args.simulate_intervals:
        print(f"Output intervals: {last_interval}")
    
    if not args.incremental_log_layers:
        break
    if reached_max_steps:
        print(f"Reached max steps at {num_logweight_steps} steps")
        break

    num_logweight_steps += 1
