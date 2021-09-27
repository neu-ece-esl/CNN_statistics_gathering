
import islpy as isl
from dataclasses import dataclass, field
from typing import (
    Tuple,
    Union,
    Set,
)
from collections import Counter
from StreamHelpers import ISLGeneratorPreprocessor


@dataclass
class IterationDomain:
    _vector: Tuple[str] = ()
    bounds: Tuple[Tuple[Union[int, str]]] = ()
    steps: Tuple[Tuple[Union[int, str]]] = ()
    parameters: Set[str] = field(default_factory=set)

    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self, vector: Tuple):
        if len(vector) != len(set(vector)):
            for iterator, count in Counter(vector).items():
                if count > 1:
                    raise SyntaxError(
                        f"Duplicate iterator {iterator} in For loops")
        for val in vector:
            if val == "_":
                raise SyntaxError(
                    "Anonymous iterators not allowed in for loops")
        self._vector = vector

    def to_abstract_repr(self, name):
        params = ISLGeneratorPreprocessor.preprocess_iterable(
            self.parameters)
        it_vector = ISLGeneratorPreprocessor.preprocess_iterable(
            self.vector)

        bound_tuples = list(
            zip(self.vector, self.bounds))
        bound_tuples_count = len(bound_tuples)

        bound_expr = ""
        for idx, (iterator, bound) in enumerate(bound_tuples):
            # upper bound
            if len(bound) == 1:
                bound_expr += f" 0 <= {iterator} < {bound[0]} "
            # upper and lower bound
            elif len(bound) == 2:
                bound_expr += f" {bound[0]} <= {iterator} < {bound[1]} "
            else:
                raise SyntaxError("Invalid number of iterator bounds")

            if idx < bound_tuples_count - 1:
                bound_expr += "and"

        step_adjustment = ""
        for idx, step in enumerate(self.steps):
            step_adjustment += "and"
            step_adjustment += f" {self.vector[idx]} mod {step} = 0 "

        structure = f"[{params}] -> {{ {name.upper()}[{it_vector}] : {bound_expr} {step_adjustment}}}"

        return structure 


@dataclass
class AccessMap:
    access_expr: str = ""
    access_expr_with_annotated_parameters: str = ""
    condition: str = ""
    condition_with_annotated_parameters: str = ""
    condition_parameters: Set[str] = field(default_factory=set)
    access_expr_parameters: Set[str] = field(default_factory=set)

    def to_abstract_repr(self, it_vector, name='', access_array_name=''):
        params = ISLGeneratorPreprocessor.preprocess_iterable(
            self.condition_parameters)
        it_vector = ISLGeneratorPreprocessor.preprocess_iterable(
            it_vector)

        access_expr = self.access_expr_with_annotated_parameters
        condition = ISLGeneratorPreprocessor.preprocess_iterable(
            self.condition)
        structure = f"[{params}]->{{ {name.upper()}[{it_vector}] -> {access_array_name}[{access_expr}] : {condition}}}"

        return structure 


@dataclass
class Invariant:
    name: str = ""

    def eval(self, args, dependent_invariants):
        #TODO: Implement
        pass
