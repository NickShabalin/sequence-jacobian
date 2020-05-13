import inspect
import re
from types import FunctionType

# noinspection PyUnresolvedReferences
import numpy as np


class VectorizedFunctionBuilder:
    # TODO: refactor; write comments
    def __init__(self, vector_arguments: dict, func: FunctionType):

        # Sources
        self._vector_arguments = vector_arguments
        self._src_func = func
        self._src_lines: list
        self._input_list: list
        self._body_lines: list
        self._output_list: list

        # Results
        self.new_input_list = []
        self.new_output_list = []
        self.new_func: FunctionType
        self._new_body_lines = []
        self._inner_vector_values = set()
        self._new_func_str: str

    def build(self):
        self._parse_source_function()
        self._build_new_input_list()
        self._build_body()
        self._compose_new_func_str()
        self._generate_new_func()

    def _parse_source_function(self):
        self._src_lines = [line.strip() for line in inspect.getsourcelines(self._src_func)[0]]
        self._input_list = inspect.signature(self._src_func).parameters.keys()
        self._body_lines = self._src_lines[2:-1]  # ignore decorator, signature and return lines
        self._output_list = self._src_lines[-1][7:].split(", ")  # ignore first 7 chars with return keyword

    def _build_new_input_list(self):
        for argument in self._input_list:
            if argument not in self._vector_arguments:
                self.new_input_list.append(argument)
                continue
            for i in range(self._vector_arguments[argument]):
                self.new_input_list.append(f"{argument}_{i + 1}")

    def _build_body(self):
        # create new body and return list
        for line in self._body_lines:
            # parse line
            line = self._distinguish_braces(line)
            left, right = line.split(" = ")
            right = right.split(" ")

            # filter vector elements in right part
            vec_el = [i for i in right if i in self._vector_arguments]
            vec_len = self._vector_arguments[vec_el[0]] if vec_el else None
            self._validate_body_line(left, vec_el, vec_len)

            # add scalar only line to new body without any changes
            if not vec_el:
                self._handle_scalars_line(line, left)
                continue

            # handle line with vector values
            self._handle_vectors_line(left, right, vec_len)

    # noinspection PyMethodMayBeStatic
    def _distinguish_braces(self, line: str) -> str:
        line = re.sub(r"\(", " ( ", line)  # surround opening braces with a whitespaces
        line = re.sub(r"\)", " ) ", line)  # surround closing braces with a whitespaces
        line = re.sub(r"\s+", " ", line)  # delete coherent whitespaces
        return line.strip()

    def _validate_body_line(self, left: str, vec_el: list, vector_len: int):
        # check if left part of the equation is not a known value
        if left in self._input_list:
            raise ValueError(f"Invalid equation")

        # check if there are no vectors of different len in same line
        if not all(self._vector_arguments[i] == vector_len for i in vec_el):
            raise ValueError("Vectors of different lengths are not allowed in the same equation")

    def _handle_scalars_line(self, line, left):
        self._new_body_lines.append(line)
        if left in self._output_list:
            self.new_output_list.append(left)

    def _handle_vectors_line(self, left: str, right: list, vec_len: int):
        for i in range(vec_len):
            line = []
            for element in right:
                if any((element in self._vector_arguments,
                        element in self._inner_vector_values)):
                    element = f"{element}_{i + 1}"
                line.append(element)

            self._new_body_lines.append(f"{left}_{i + 1} = {' '.join(line)}")

            if left in self._output_list:
                self.new_output_list.append(f"{left}_{i + 1}")
                self._inner_vector_values.add(left)
            else:
                self._inner_vector_values.add(left)

    def _compose_new_func_str(self):
        # delete unwanted spaces near braces
        self._new_body_lines = [re.sub(r"(?<=[a-zA-Z\d(])\s\(", "(", i) for i in self._new_body_lines]
        self._new_body_lines = [re.sub(r"\)\s(?=[a-zA-Z\d)])", ")", i) for i in self._new_body_lines]
        self._new_body_lines = [re.sub(r"\(\s", "(", i) for i in self._new_body_lines]
        self._new_body_lines = [re.sub(r"\s\)", ")", i) for i in self._new_body_lines]

        # compose list of lines
        signature_line = f"def {self._src_func.__name__}({', '.join(self.new_input_list)}):"
        return_line = "return " + ", ".join(self.new_output_list)
        new_function_lines = [signature_line] + self._new_body_lines + [return_line]

        # add idents to lines (except the first line)
        for i in range(1, len(new_function_lines)):
            new_function_lines[i] = " " * 4 + new_function_lines[i]

        # add line breaks (except the last line)
        for vector_number in range(len(new_function_lines) - 1):
            new_function_lines[vector_number] += "\n"

        self._new_func_str = "".join(new_function_lines)
        print("-" * 20)
        print(self._new_func_str)

    def _generate_new_func(self):
        code = compile(self._new_func_str, '<string>', 'exec')
        self.new_func = FunctionType(code.co_consts[0], globals(), self._src_func.__name__)
