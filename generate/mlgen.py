from dataclasses import dataclass
from typing import Optional
import warnings
import numpy as np
import math

import fp as fp

import hdlgen


def make_tree_adder(float_environment, on_module, summing_wires, out_wire, width):
    if len(summing_wires) == 0: raise ValueError("No wires provided")

    if len(summing_wires) == 1: return on_module.AddAssignment(out_wire, summing_wires[0])

    # Otherwise...
    branch_l = on_module.AddWire(width, "branch_left")
    branch_r = on_module.AddWire(width, "branch_right")

    make_tree_adder(float_environment, on_module, summing_wires[:len(summing_wires)//2], branch_l, width)

    make_tree_adder(float_environment, on_module, summing_wires[len(summing_wires)//2:], branch_r, width)

    float_environment.add_ip(on_module, "fp_adder", {"argumenta": branch_l, "argumentb": branch_r, "out": out_wire})

def make_linear_adder(float_environment, on_module, summing_wires, out_wire, width):
    if len(summing_wires) == 0: raise ValueError("No wires provided")

    if len(summing_wires) == 1: return on_module.AddAssignment(out_wire, summing_wires[0])

    last_sum_wire = summing_wires[0]

    for wire in summing_wires[1:-1]: # Omit first and last elements
        add_stage = on_module.AddWire(width, "add_stage")
        float_environment.add_ip(on_module, "fp_adder", {"argumenta": last_sum_wire, "argumentb": wire, "out": add_stage})
        last_sum_wire = add_stage
# Otherwise...
    float_environment.add_ip(on_module, "fp_adder", {"argumenta": last_sum_wire, "argumentb": summing_wires[-1], "out": out_wire})
    
def make_aio_adder(float_environment, on_module: hdlgen.Module, summing_wires: list[hdlgen.Wire], out_wire: hdlgen.Wire, width):
    if len(summing_wires) == 0: raise ValueError("No wires provided")

    arg_wire = on_module.AddWire(width, "summer_arguments", length=len(summing_wires))


    float_environment.add_ip(on_module, "fp_sum", {"argument_array": arg_wire, "out": out_wire}, {"inputcount": len(summing_wires)})

    on_module.AddAssignment(arg_wire, hdlgen.Concatenation(summing_wires))

def make_neuron_stage(float_environment, on_module, input_wires, weights, out_wire, width):
    collected_multiplication_wires = []

    for (in_wire, weight) in zip(input_wires, weights, strict=True):
        multiply_out = on_module.AddWire(width, "mult_out")
        #multiplicand_param = "16'h" + struct.pack('>f', weight).hex()
        #on_module.AddExternalModule("fp_multiplier", {"argumenta": in_wire, "out": multiply_out}, {"multiplicand": multiplicand_param})
        
        float_environment.add_ip(on_module, "fp_multiplybypowerof2", {"argumenta": in_wire, "out": multiply_out}, {"power": int(weight)})
        collected_multiplication_wires.append(multiply_out)

    make_linear_adder(float_environment, on_module, collected_multiplication_wires, out_wire, width)

# pytorch is prone to API changes and extracting model params is generally poorly-documented
# + a bit buggy so we make our own scheme here
    
class SequentialStepHDL:
    
    def apply(self, previous_neuron_buses: list[hdlgen.Wire], target_module: hdlgen.Module, float_environment: fp.FloatEnvironment) -> list[hdlgen.Wire]:
        raise NotImplementedError()

    def eval(self, inputs: list[float]):
        raise NotImplementedError()

class ActivationStep(SequentialStepHDL):
    pass

class ReLUStep(ActivationStep):
    def apply(self, previous_neuron_buses, target_module, float_environment: fp.FloatEnvironment):
        out_layer = [] 
        for neuron_in in previous_neuron_buses:
            neuron_out = target_module.AddWire(float_environment.float_size, f"neuron_relu")
            out_layer.append(neuron_out)

            float_environment.add_ip(target_module,
                "fp_activation_relu",
                {"argumenta": neuron_in, "out": neuron_out})
        
        return out_layer

    def eval(self, inputs):
        return [x if x>0 else 0 for x in inputs]
    
class BiasStep(SequentialStepHDL):
    def __init__(self, biases: list[int]):
        self.biases = biases

    def apply(self, previous_neuron_buses: list[hdlgen.Wire], target_module: hdlgen.Module, float_environment: fp.FloatEnvironment):
        out = []
        
        for (neuron, bias) in zip(previous_neuron_buses, self.biases, strict=True):
            out_neuron = target_module.AddWire(float_environment.float_size, "biased")
            bias_as_literal = hdlgen.AutoSizeLiteral(float_environment.float_to_hexstring(bias), "hex")

            float_environment.add_ip(target_module, "fp_adder", {"argumenta": neuron, "argumentb": bias_as_literal, "out": out_neuron})

            out.append(out_neuron)

        return out
    
    def eval(self, inputs):
        return [x + bias for (x, bias) in zip(inputs, self.biases, strict=True)]

class DenseLogLayer(SequentialStepHDL):
    def __init__(self, log_weights_signs : list[list[list[list[float]]]]):
        self.log_weights_signs = log_weights_signs
    
    def apply(self, previous_neuron_buses: list[hdlgen.Wire], target_module: hdlgen.Module, float_environment: fp.FloatEnvironment):
        if len(previous_neuron_buses) != len(self.log_weights_signs[0]):
            raise ValueError(f"Size mismatch: given {len(previous_neuron_buses)} wires for a {len(self.log_weights_signs)}x{len(self.log_weights_signs[0])} matrix.")

        post_mul_neurons = [target_module.AddWire(float_environment.float_size, f"neuron_multiplied_{i}") for i in range(len(self.log_weights_signs))] 
        print("Made target array of size", len(post_mul_neurons))
        for target_neuron, neuron_log_weights_signs in zip(post_mul_neurons, self.log_weights_signs, strict=True):
                collected_multiplication_wires = []
                for (in_wire, this_connection_weights_signs) in zip(previous_neuron_buses, neuron_log_weights_signs, strict=True):
                    for (log_weight, negate) in this_connection_weights_signs:
                        multiply_out = target_module.AddWire(float_environment.float_size, "mult_out")
                        
                        float_environment.add_ip(target_module, "fp_multiplybypowerof2", {"argumenta": in_wire, "out": multiply_out}, {"power": log_weight, "negate": int(negate)})
                        collected_multiplication_wires.append(multiply_out)
                if len(collected_multiplication_wires) == 0:
                    warnings.warn("Neuron had no contributing neurons within precision!")
                    target_module.AddAssignment(target_neuron, hdlgen.AutoSizeLiteral(0))
                else:
                    make_aio_adder(float_environment, target_module, collected_multiplication_wires, target_neuron, float_environment.float_size)

        return post_mul_neurons

    def eval(self, inputs):
        return [
            sum(
                sum(
                    (-1 if negative else 1) * x * math.pow(2,weight) for (weight, negative) in neuron_log_weights_signs
                ) for neuron_log_weights_signs, x in zip(this_output_log_weights_signs, inputs, strict=True)
            ) for this_output_log_weights_signs in self.log_weights_signs
        ]
    
class DenseLayer(SequentialStepHDL):
    def __init__(self, weights : list[list[float]]):
        self.weights = weights
    
    def eval(self, inputs):
        return [
            sum(
                x * weight for weight, x in zip(neuron_weights, inputs)
            ) for neuron_weights in self.weights
        ]
@dataclass
class Model:
    layers: list[SequentialStepHDL]
    input_count: int
    output_count: int