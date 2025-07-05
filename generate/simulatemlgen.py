import math
import pickle
from typing import Iterable
import fp as fp
import warnings
import conversionutils
import mlgen
import argparse, os

from dataclasses import dataclass

parser = argparse.ArgumentParser(description="Simulate the given mlgen model")
parser.add_argument("model", help="the input .mlgen array")
parser.add_argument("--log-incremental-inputs", "-inc", action="store_true", help="Simulate feeding inputs into the model incrementally, showing model output at each value")
parser.add_argument("--log-incremental-steps", "-inc-n", type=int, default=10, help="Total number of log-incremental input steps to simulate")

parser.add_argument("--simulate-hyperbox-interval", "-shint", action="store_true")
parser.add_argument("--hyperbox-artificial-interval", "-ahint", type=float, default=0.0, help="Introduce artificial interval to network inputs of +/- this number")

parser.add_argument("--simulate-polytope-interval", "-spint", action="store_true")
parser.add_argument("--ignore-layer-quantization-loss", "-nofloss", action="store_true", help="Ignore layer computation inaccuracy impact on bounds")
#parser.add_argument("input", help="comma-separated list of inputs")

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

@dataclass
class SimulationIterationOutput:
    hyperbox_intervals: list[list[tuple[float, float]]]
    layer_values: list[list[float]]


def simulate(mlgen_model: mlgen.Model, input: list, do_log_incremental_inputs: bool, log_incremental_steps: int, ignore_layer_quantization_loss:bool=False, hyperbox_artificial_interval: float=0,log_messages=False) -> Iterable[SimulationIterationOutput]:
    
    num_logfrag_steps = 1

    

    # generate log-fragment inputs
    if do_log_incremental_inputs:
        inputs_fragments = [ conversionutils.make_log_mult(x, frag_count=log_incremental_steps) for x in input]
    else:
        # stupid python typing
        inputs_fragments = []

    incremental_steps = 1

    while True:
        if do_log_incremental_inputs:
            last_layer = [
                    sum(
                            math.pow(2.0, weight) * (-1.0 if negative else 1.0)
                        for index, (weight, negative) in enumerate(input_fragments) if index < incremental_steps
                    )
                for input_fragments in inputs_fragments
            ]


            # Input error is half the most-recently-incrementally-added input weight, since that's the
            # sum of all possible subsequent weights. If incremental steps has reached the number of fragments
            # then input error is 0.

            poss_input_errors = [

                    math.pow(2.0, input_fragments[incremental_steps - 1][0] - 1) if incremental_steps <= len(input_fragments) else 0.0
                for input_fragments in inputs_fragments
            ]

            if log_messages: print(f"Incremental input layer is {last_layer}")

            last_hyperbox_interval = [(x - poss_input_err - hyperbox_artificial_interval, x + poss_input_err + hyperbox_artificial_interval) for x, poss_input_err in zip(last_layer, poss_input_errors)]
            if log_messages:
                print(f"  - got input intervals {abbreviate_list(last_hyperbox_interval)}")
            
        else:
            # just use the exact input
            last_layer = input
            last_hyperbox_interval = [(x - hyperbox_artificial_interval, x + hyperbox_artificial_interval) for x in last_layer]

        if do_log_incremental_inputs:
            print(f"Simulating with {num_logfrag_steps} input log-fragment steps")
        reached_max_steps = True
        
        iteration_layer_values = []
        iteration_hyperbox_intervals = []

        for layer_index, layer in enumerate(mlgen_model.layers):
            #disabled because stupid and broken
            """if isinstance(layer, mlgen.WeightIncrementalLogLayer):
                if args.log_incremental_inputs:
                    reached_max_steps = reached_max_steps and layer.set_steps(num_logfrag_steps)
                else:
                    warnings.warn("Encountered incremental log layer but not advancing (use --incremental-log-layers)!")"""
            last_layer = layer.eval(last_layer)
            iteration_layer_values.append(last_layer)
            #print(f"- Layer #{layer_index} ({layer}) got {abbreviate_list(last_layer)}")

            #if args.simulate_hyperbox_interval:
            last_hyperbox_interval = layer.eval_axishyperbox_domain(last_hyperbox_interval, ignore_layer_loss=ignore_layer_quantization_loss)
            iteration_hyperbox_intervals.append(last_hyperbox_interval)
            #    print(f"  - got{' (lossless)' if args.ignore_layer_quantization_loss else ''} intervals {abbreviate_list(last_hyperbox_interval)}")

        print(f"Output: {last_layer}")
        if log_messages:
            print(f"Output intervals: {last_hyperbox_interval}")
        
        yield SimulationIterationOutput(
            hyperbox_intervals=iteration_hyperbox_intervals,
            layer_values=iteration_layer_values
        )

        if not do_log_incremental_inputs:
            break
        #if reached_max_steps:
        #    print(f"Reached max steps at {num_logfrag_steps} steps")
        #    break
        if incremental_steps >= log_incremental_steps:
            break

        num_logfrag_steps += 1
        incremental_steps += 1
        
def hyperbox_has_onehot_winner(hyperbox: list[tuple[float, float]], higher_better=True) -> int|None:
    # Determine if any of the elements of the onehot encoded "outcome" will definitely "win" against all others
    # equivalent to finding if the highest minimum is higher than all the other maximums (for higher_better case)

    # messy
    best_worsebound_index = (max if higher_better else min)(range(len(hyperbox)), key=lambda index: hyperbox[index][0 if higher_better else 1])
    best_worsebound = hyperbox[best_worsebound_index][0 if higher_better else 1]

    best_other_betterbound = (max if higher_better else min)(hyperbox[i][1 if higher_better else 0] for i in range(len(hyperbox)) if i != best_worsebound_index)

    if higher_better:
        if best_worsebound > best_other_betterbound:
            return best_worsebound_index
    else:
        if best_worsebound < best_other_betterbound:
            return best_worsebound_index
    
    return None

if __name__ == "__main__":

    args = parser.parse_args()

    f = open(args.model, "rb")
    mlgen_model: mlgen.Model = pickle.load(f)
    f.close()


    input = [-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.258611,1.026908,1.026908,1.026908,-0.105876,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.835989,2.490618,2.796088,2.808815,2.796088,1.128731,0.543247,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.708710,2.096053,2.643353,2.796088,2.796088,2.490618,2.312427,2.796088,1.510569,-0.093148,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.182244,0.950541,2.223332,2.796088,2.719720,2.236060,0.645071,0.174138,0.543247,2.719720,2.554257,0.046859,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,1.726943,2.796088,2.808815,2.796088,1.077820,-0.424074,-0.424074,-0.424074,-0.424074,1.103275,2.643353,1.370562,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.670527,2.808815,2.808815,2.821543,1.383290,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,1.306922,2.808815,1.803311,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.059587,2.376067,2.796088,2.796088,2.197876,-0.220427,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,1.905134,2.796088,1.192371,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.334979,0.301417,1.408745,2.796088,2.630625,1.879678,-0.105876,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.182244,2.248787,2.719720,0.683254,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.148682,2.796088,2.808815,2.681536,1.790583,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.263233,2.401522,2.236060,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,1.739671,2.796088,2.821543,2.325155,-0.029509,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,2.452434,2.796088,1.637848,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.301417,2.808815,2.808815,1.383290,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,2.032413,2.808815,2.096053,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.054964,2.134236,2.796088,2.796088,0.785078,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.296795,0.657799,2.617897,2.796088,0.479608,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,2.057869,2.796088,2.719720,1.077820,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.339601,2.796088,2.821543,1.548752,-0.258611,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.390512,2.643353,2.796088,2.439706,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.059587,2.452434,2.796088,1.854222,-0.309523,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.390512,2.643353,2.796088,1.434201,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.186866,1.688759,2.592441,1.001452,0.377784,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,1.026908,2.821543,2.490618,0.174138,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,1.828766,2.808815,2.490618,0.174138,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,1.026908,2.796088,2.312427,-0.054964,-0.424074,-0.424074,-0.054964,0.657799,0.861445,2.452434,2.808815,2.439706,0.237777,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,1.408745,2.796088,2.796088,2.325155,2.096053,2.108780,2.325155,2.796088,2.719720,2.070597,1.281466,0.046859,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.657799,2.363339,2.796088,2.796088,2.796088,2.808815,2.681536,2.439706,2.121508,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,0.059587,1.001452,1.001452,1.001452,1.014180,0.530519,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074,-0.424074]
    
    last_layer = []
    
    for iteration_index, iteration_output in enumerate(simulate(mlgen_model, input, do_log_incremental_inputs=args.log_incremental_inputs, log_incremental_steps=args.log_incremental_steps, ignore_layer_quantization_loss=args.ignore_layer_quantization_loss)):
        last_layer = []
        last_hyperbox = []
        for layer_index, (layer, values, hyperbox) in enumerate(zip(mlgen_model.layers, iteration_output.layer_values, iteration_output.hyperbox_intervals, strict=True)):
            print(f" - Layer #{layer_index} ({layer}) got {abbreviate_list(values)}")
            print(f"   (hyperbox): {abbreviate_list(hyperbox)}")
            last_layer = values
            last_hyperbox = hyperbox

        print("Output:", last_layer)
        print("Hyperbox:", last_hyperbox)
        winner = hyperbox_has_onehot_winner(last_hyperbox) 
        if winner is not None:
            print(f"Winner: {winner} ({last_hyperbox[winner]})")
            print("Stopped at iteration", iteration_index, "because hyperbox unambiguous already")
            break
