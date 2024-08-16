from dataclasses import dataclass
from typing import Optional, NamedTuple
import warnings
import numpy as np
import math

import fp as fp

import hdlgen


class WeightFragment(NamedTuple):
    exponent: int
    negative: bool


def make_tree_adder(float_environment, on_module, summing_wires, out_wire, width):
    if len(summing_wires) == 0:
        raise ValueError("No wires provided")

    if len(summing_wires) == 1:
        return on_module.AddAssignment(out_wire, summing_wires[0])

    # Otherwise...
    branch_l = on_module.AddWire(width, "branch_left")
    branch_r = on_module.AddWire(width, "branch_right")

    make_tree_adder(
        float_environment,
        on_module,
        summing_wires[: len(summing_wires) // 2],
        branch_l,
        width,
    )

    make_tree_adder(
        float_environment,
        on_module,
        summing_wires[len(summing_wires) // 2 :],
        branch_r,
        width,
    )

    float_environment.add_ip(
        on_module,
        "fp_adder",
        {"argumenta": branch_l, "argumentb": branch_r, "out": out_wire},
    )


def make_linear_adder(float_environment, on_module, summing_wires, out_wire, width):
    if len(summing_wires) == 0:
        raise ValueError("No wires provided")

    if len(summing_wires) == 1:
        return on_module.AddAssignment(out_wire, summing_wires[0])

    last_sum_wire = summing_wires[0]

    for wire in summing_wires[1:-1]:  # Omit first and last elements
        add_stage = on_module.AddWire(width, "add_stage")
        float_environment.add_ip(
            on_module,
            "fp_adder",
            {"argumenta": last_sum_wire, "argumentb": wire, "out": add_stage},
        )
        last_sum_wire = add_stage
    # Otherwise...
    float_environment.add_ip(
        on_module,
        "fp_adder",
        {"argumenta": last_sum_wire, "argumentb": summing_wires[-1], "out": out_wire},
    )


def make_aio_adder(
    float_environment,
    on_module: hdlgen.Module,
    summing_wires: list[hdlgen.Wire],
    out_wire: hdlgen.Wire,
    width,
):
    if len(summing_wires) == 0:
        raise ValueError("No wires provided")

    arg_wire = on_module.AddWire(width, "summer_arguments", length=len(summing_wires))

    float_environment.add_ip(
        on_module,
        "fp_sum",
        {"argument_array": arg_wire, "out": out_wire},
        {"inputcount": len(summing_wires)},
    )

    on_module.AddAssignment(arg_wire, hdlgen.Concatenation(summing_wires))


def make_neuron_stage(
    float_environment, on_module, input_wires, weights, out_wire, width
):
    collected_multiplication_wires = []

    for in_wire, weight in zip(input_wires, weights, strict=True):
        multiply_out = on_module.AddWire(width, "mult_out")
        # multiplicand_param = "16'h" + struct.pack('>f', weight).hex()
        # on_module.AddExternalModule("fp_multiplier", {"argumenta": in_wire, "out": multiply_out}, {"multiplicand": multiplicand_param})

        float_environment.add_ip(
            on_module,
            "fp_multiplybypowerof2",
            {"argumenta": in_wire, "out": multiply_out},
            {"power": int(weight)},
        )
        collected_multiplication_wires.append(multiply_out)

    make_linear_adder(
        float_environment, on_module, collected_multiplication_wires, out_wire, width
    )


# pytorch is prone to API changes and extracting model params is generally poorly-documented
# + a bit buggy so we make our own scheme here


def sort_tuple(tup: tuple[float, float]) -> tuple[float, float]:
    "No generics in Python yet so this is a bit ugly"
    if tup[0] > tup[1]:
        return (tup[1], tup[0])
    else:
        return (tup[0], tup[1])


class SequentialStepHDL:

    def apply(
        self,
        previous_neuron_buses: list[hdlgen.Wire],
        target_module: hdlgen.Module,
        float_environment: fp.FloatEnvironment,
    ) -> list[hdlgen.Wire]:
        raise NotImplementedError()

    def eval(self, vector_in: list[float]):
        raise NotImplementedError()

    def eval_interval(
        self, intervals_vector_in: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Return a lower and upper bound on layer output, given the lower and upper bound layer
        inputs."""
        raise NotImplementedError(
            "For a monotonic layer, inherit from Monotoni cStep for an implementation"
        )


class MonotonicStep(SequentialStepHDL):
    def eval_interval(
        self, intervals_vector_in: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """This implementation returns an interval of vectors based on the "actual"
        output of the layer against the two bounds layer inputs. This may not be appropriate for
        non-monotonic layers."""
        lower_bounds = self.eval([t[0] for t in intervals_vector_in])
        upper_bounds = self.eval([t[1] for t in intervals_vector_in])

        return [sort_tuple((x, y)) for x, y in zip(lower_bounds, upper_bounds)]


class ActivationStep(SequentialStepHDL):
    pass


class ReLUStep(MonotonicStep, ActivationStep):
    def apply(
        self,
        previous_neuron_buses,
        target_module,
        float_environment: fp.FloatEnvironment,
    ):
        out_layer = []
        for neuron_in in previous_neuron_buses:
            neuron_out = target_module.AddWire(
                float_environment.float_size, f"neuron_relu"
            )
            out_layer.append(neuron_out)

            float_environment.add_ip(
                target_module,
                "fp_activation_relu",
                {"argumenta": neuron_in, "out": neuron_out},
            )

        return out_layer

    def eval(self, vector_in):
        return [x if x > 0 else 0 for x in vector_in]


class BiasStep(MonotonicStep, SequentialStepHDL):
    def __init__(self, biases: list[int]):
        self.biases = biases

    def apply(
        self,
        previous_neuron_buses: list[hdlgen.Wire],
        target_module: hdlgen.Module,
        float_environment: fp.FloatEnvironment,
    ):
        out = []

        for neuron, bias in zip(previous_neuron_buses, self.biases, strict=True):
            out_neuron = target_module.AddWire(float_environment.float_size, "biased")
            bias_as_literal = hdlgen.AutoSizeLiteral(
                float_environment.float_to_hexstring(bias), "hex"
            )

            float_environment.add_ip(
                target_module,
                "fp_adder",
                {"argumenta": neuron, "argumentb": bias_as_literal, "out": out_neuron},
            )

            out.append(out_neuron)

        return out

    def eval(self, vector_in):
        return [x + bias for (x, bias) in zip(vector_in, self.biases, strict=True)]


class DenseLogLayer(SequentialStepHDL):
    def __init__(self, weight_fragments: list[list[list[WeightFragment]]]):
        self.weight_fragments = weight_fragments

    def apply(
        self,
        previous_neuron_buses: list[hdlgen.Wire],
        target_module: hdlgen.Module,
        float_environment: fp.FloatEnvironment,
    ):
        if len(previous_neuron_buses) != len(self.weight_fragments[0]):
            raise ValueError(
                f"Size mismatch: given {len(previous_neuron_buses)} wires for a {len(self.weight_fragments)}x{len(self.weight_fragments[0])} matrix."
            )

        post_mul_neurons = [
            target_module.AddWire(
                float_environment.float_size, f"neuron_multiplied_{i}"
            )
            for i in range(len(self.weight_fragments))
        ]
        print("Made target array of size", len(post_mul_neurons))
        for target_neuron, neuron_log_weights_signs in zip(
            post_mul_neurons, self.weight_fragments, strict=True
        ):
            collected_multiplication_wires = []
            for in_wire, this_connection_weights_signs in zip(
                previous_neuron_buses, neuron_log_weights_signs, strict=True
            ):
                for exponent, negate in this_connection_weights_signs:
                    multiply_out = target_module.AddWire(
                        float_environment.float_size, "mult_out"
                    )

                    float_environment.add_ip(
                        target_module,
                        "fp_multiplybypowerof2",
                        {"argumenta": in_wire, "out": multiply_out},
                        {"power": exponent, "negate": int(negate)},
                    )
                    collected_multiplication_wires.append(multiply_out)
            if len(collected_multiplication_wires) == 0:
                warnings.warn("Neuron had no contributing neurons within precision!")
                target_module.AddAssignment(target_neuron, hdlgen.AutoSizeLiteral(0))
            else:
                make_aio_adder(
                    float_environment,
                    target_module,
                    collected_multiplication_wires,
                    target_neuron,
                    float_environment.float_size,
                )

        return post_mul_neurons

    def eval(self, vector_in):
        return [
            sum(
                sum(
                    (-1.0 if negative else 1.0) * x * math.pow(2, weight)
                    for (weight, negative) in neuron_log_weights_signs
                )
                for neuron_log_weights_signs, x in zip(
                    this_output_log_weights_signs, vector_in, strict=True
                )
            )
            for this_output_log_weights_signs in self.weight_fragments
        ]


class IncrementalLogLayer(DenseLogLayer):
    """Doesnt actually incrementally compute accumulated values, just computes outputs
    in one go upon eval() for flexibility"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_num_weights = max(
            (max((len(y) for y in x), default=0) for x in self.weight_fragments),
            default=0,
        )
        print(f"Getting max weight count of {self.max_num_weights}")
        self.use_num_weights = 1

    def apply(self, *args, **kwargs):
        raise NotImplementedError()

    def set_steps(self, steps):
        self.use_num_weights = steps
        return self.use_num_weights == self.max_num_weights

    def eval_interval(
        self, intervals_vector_in: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Works by finding the current output interval using only the weight-fragments
        being used so far (use_num_weights) plus maximum positive or negative deviation,
        whose absolute value is given by the sum of all possible remaining weight-
        fragments"""

        intervals = []

        for weight_fragments_row in self.weight_fragments:
            # Find most negative/most positive possible contributions to this output neuron
            min_sum = 0.0
            max_sum = 0.0
            for weight_fragments, (interval_lower, interval_higher) in zip(
                weight_fragments_row, intervals_vector_in, strict=True
            ):
                current_weight = sum(
                    (-1.0 if negative else 1.0) * math.pow(2, weight)
                    for (index, (weight, negative)) in enumerate(weight_fragments)
                    if index < self.use_num_weights
                )

                """Max contribution from unprocessed (as yet "unknown") fragments is
                + or - sum of all powers of 2 less than the smallest current used power:
                this is equal to the magnitude of the smallest current used power"""
                
                unused_fragments_contrib = (
                    math.pow(2, weight_fragments[self.use_num_weights-1].exponent)
                ) if (self.use_num_weights > 0) and (self.use_num_weights <= len(weight_fragments)) else 0.0

                max_sum += (
                    current_weight
                    + (
                        unused_fragments_contrib
                        if interval_higher > 0
                        else -unused_fragments_contrib
                    )
                ) * interval_higher
                min_sum += (
                    current_weight
                    + (
                        -unused_fragments_contrib
                        if interval_lower > 0
                        else unused_fragments_contrib
                    )
                ) * interval_lower

            intervals.append((min_sum, max_sum))

        return intervals

    def eval(self, vector_in) -> list[float]:
        return [
            sum(
                sum(
                    (-1.0 if negative else 1.0) * x * math.pow(2, weight)
                    for (index, (weight, negative)) in enumerate(weight_fragments)
                    if index < self.use_num_weights
                )
                for weight_fragments, x in zip(
                    this_output_log_weights_signs, vector_in, strict=True
                )
            )
            for this_output_log_weights_signs in self.weight_fragments
        ]


class DenseLayer(MonotonicStep, SequentialStepHDL):
    def __init__(self, weights: list[list[float]]):
        self.weights = weights

    def eval_interval(
        self, intervals_vector_in: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """For each output, determine the interval bound: the minimum
        will be the sum of the most-negative-possible contributions, and
        the maximum will be the sum of the most-positive-possible
        contributions from each input in the output's column"""

        intervals = []

        for weight_row in self.weights:
            # Find most negative/most positive possible contributions to this output neuron
            min_sum = 0
            max_sum = 0
            for weight, (interval_lower, interval_higher) in zip(
                weight_row, intervals_vector_in, strict=True
            ):
                max_sum += weight * (interval_higher if weight > 0 else interval_lower)
                min_sum += weight * (interval_lower if weight > 0 else interval_higher)

            intervals.append((min_sum, max_sum))

        return intervals

    def eval(self, vector_in):
        return [
            sum(x * weight for weight, x in zip(neuron_weights, vector_in, strict=True))
            for neuron_weights in self.weights
        ]


@dataclass
class Model:
    layers: list[SequentialStepHDL]
    input_count: int
    output_count: int
