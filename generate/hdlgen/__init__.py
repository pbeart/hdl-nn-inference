from enum import Enum
from hdlgen import helpers


WireType = Enum('WireType', ['Wire', 'Input', 'Output'])

class Expression:
    def hdl(self):
        raise NotImplementedError()
    def hdl_expression(self):
        raise NotImplementedError()
    
class AutoSizeLiteral(Expression):
    def __init__(self, value: int|str, display_format="hex"):
        self.value = value
        self.display_format = display_format

        if not display_format in ["hex", "dec"]:
            raise NotImplementedError(f"Can't understand literal display format `{display_format}`")
        
    def hdl(self):
        return ""
    
    def hdl_expression(self):
        if isinstance(self.value, int):
            representation = f"{self.value:x}"
            if self.display_format == "hex":
                representation = f"{self.value:x}"
            elif self.display_format == "dec":
                representation = f"{self.value}"
            else:
                raise NotImplementedError()
        else:
            representation = self.value

        if self.display_format == "hex":
            return f"'h{representation}"
        elif self.display_format == "dec":
            return f"'d{representation}"
        else:
            raise NotImplementedError()


class Wire(Expression):
    def __init__(self, id, size, wiretype, module, length=None):
        self.parent = module
        self.size = size
        self.id = id
        self.type = wiretype
        self.length = length

    def __eq__(self, value):
        if isinstance(value, self.__class__):
            return self.id == value.id
        else:
            raise TypeError(f"Cannot compare {self.__class__} with {type(value)}")
        
    def __str__(self):
        return f"<hdlgen.Wire '{self.id}' ([{self.size-1}:0], {self.length} elements)>"

    def hdl(self):
        if self.size == 1:
            size_string = ""
        else:
            size_string = f"[{self.size-1}:0] "

        if self.length == None:
            length_string = ""
        else:
            length_string = f" [{self.length}]"

        if self.type == WireType.Input:
            declaration = "input"
        elif self.type == WireType.Output:
            declaration = "output"
        else:
            declaration = "wire"

        return f"{declaration} {size_string}{self.id}{length_string};\n"

    def hdl_expression(self):
        return self.id

class Concatenation(Expression):
    def __init__(self, elements):
        self.elements = elements

    def hdl(self):
        return "" # Existence of concatenation doesn't implicitly need any other HDL to be generated

    def hdl_expression(self):
        return "'{" + ", ".join(element.hdl_expression() for element in self.elements) + "}"
    
class Indexing(Expression):
    def __init__(self, of_what, index):
        self.of_what = of_what
        self.index = index
    
    def hdl(self):
        return ""
    
    def hdl_expression(self):
        return f"{self.of_what.hdl_expression()}[{self.index}]"

class HDL:
    def hdl(self):
        raise NotImplementedError

class Assignment(HDL):
    def __init__(self, target, source):
        self.target = target
        self.source = source

    def hdl(self):
        return f"assign {self.target.id} = {self.source.hdl_expression()};\n"
    
class Module(HDL):
    def __init__(self, id):
        self.id = id
        self.wire_counter = 0
        self.module_counter = 0
        self.wires: list[Wire] = []
        self.inputs = []
        self.outputs = []
        self.assignments: list[Assignment] = []
        self.external_modules = []

    def _make_wire(self, name, size, length=None):
        w = Wire(f"w_{self.wire_counter}{'' if name is None else '_' + name}", size, WireType.Wire, self, length)
        self.wire_counter += 1
        return w
    
    def _make_input_output(self, id: str, size, type, length: int|None):
        return Wire(id, size, type, self, length)

    def AddWire(self, size: int = 1, name:str|None = None, length:int|None =None):
        w = self._make_wire(name, size, length=length)
        self.wires.append(w)
        return w
    
    def AddInput(self, id: str, size: int = 1, length: int|None = None):
        i = self._make_input_output(id, size, WireType.Input, length=length)
        self.inputs.append(i)
        return i
    
    def AddOutput(self, id: str, size: int = 1, length: int|None = None):
        o = self._make_input_output(id, size, WireType.Output, length=length)
        self.outputs.append(o)
        return o
    
    def AddExternalModule(self, module_name, connections, parameters={}):
        m = ExternalModule(f"module_{self.module_counter}_{module_name}", self, module_name, connections, parameters)
        self.module_counter += 1
        self.external_modules.append(m)
        return m
    
    def AddAssignment(self, target: Expression, source_wire):
        self.assignments.append(Assignment(target, source_wire))
    
    def checks(self):
        
        # todo: can't handle assignments via wire connections on modules
        """for wire in self.wires:
            found = False
            for assignment in self.assignments:
                if wire == assignment.target:
                    if found:
                        raise ValueError(f"Wire `{wire.hdl().strip()}` assigned to multiple times!")
                    found = True

            if not found:
                raise ValueError(f"Wire `{wire.hdl().strip()}` has no driver!")"""
        
        # cant deal with concatenations
        """for assignment in self.assignments:
            found_target = False
            found_source = False
            for wire in self.wires:
                if wire == assignment.target:
                    found_target = True
                elif wire == assignment.source:
                    found_source = True
                
            
            if not found_target:
                raise ValueError(f"Assignment `{assignment.hdl().strip()}` has a nonexistent target wire!")
            
            if not found_source:
                raise ValueError(f"Assignment `{assignment.hdl().strip()}` has a nonexistent source wire!")"""
    
            

    def hdl(self):
        o = f"module {self.id}("

        input_ids = [input.id for input in self.inputs]
        output_ids = [output.id for output in self.outputs]
        
        o += ", ".join([*input_ids, *output_ids])
        o += ");\n"

        body = ""
        for input in self.inputs:
            body += input.hdl()

        for output in self.outputs:
            body += output.hdl()

        for wire in self.wires:
            body += wire.hdl()

        for external_module in self.external_modules:
            body += external_module.hdl()

        for assignment in self.assignments:
            body += assignment.hdl()

        body = helpers.indent(body, 1)

        o += body

        o += "endmodule\n"

        return o

class ExternalModule(HDL):
    def __init__(self, id, module, module_name, connections, parameters):
        self.module_name = module_name
        self.parent = module
        self.id = id
        self.connections = connections
        self.parameters = parameters
    
    def hdl(self):
        
        connections = [f".{inner}({outer.hdl_expression()})" for (inner, outer) in self.connections.items()]

        for outer in self.connections.values():
            if isinstance(outer, Wire):
                assert outer.parent == self.parent

        parameters = [f".{inner}({outer})" for (inner, outer) in self.parameters.items()]

        if len(self.parameters) == 0:
            parameter_string = ""
        else:
            parameter_string = "#(" + f",\n{helpers.indentation}".join(parameters) + ") "

        out = f"{self.module_name} {parameter_string}{self.id}(\n"
        out += helpers.indentation + f",\n{helpers.indentation}".join(connections)
        out += ");\n"

        return out
    
