from dataclasses import dataclass
from decimal import Decimal
import struct

@dataclass
class FloatDefinition:
    float_size: int
    exponent_size: int
    struct_format: str

# https://docs.python.org/3/library/struct.html#format-characters

# todo: probably better if float definitions are instead premade FloatEnvironments which you can pick from

FLOAT_DEFINITIONS = {
    "binary16": FloatDefinition(16, 5, "e"),
    #"bfloat16": FloatDefinition(16, 8),
    "binary32": FloatDefinition(32, 8, "f"),
    "binary64": FloatDefinition(64, 11, "d"),
}


class FloatEnvironment:
    def __init__(self, float_type="binary16"):
        float_info = FLOAT_DEFINITIONS[float_type]
        self.float_info = float_info
        self.float_size = float_info.float_size
        self.exponent_size = float_info.exponent_size
        self.significand_size = float_info.float_size - float_info.exponent_size - 1
        self.base_parameters = {"floatsize": float_info.float_size, "exponentsize": float_info.exponent_size}

    def add_ip(self, module, ip_name, connections, parameters={}):
        module.AddExternalModule(ip_name,
            connections,
            dict(self.base_parameters, **parameters))
        
    # Thanks to https://stackoverflow.com/questions/16444726/binary-representation-of-float-in-python-bits-not-hex
    def float_to_hexstring(self, value):
        return ''.join('{:0>2x}'.format(c) for c in struct.pack(f'!{self.float_info.struct_format}', value))