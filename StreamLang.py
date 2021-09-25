# To add a new cell, type ''
# To add a new markdown cell, type ' [markdown]'


from ast import Eq
import islpy as isl
from myhdl import (
    block,
    delay,
    always_seq,
    instance,
    always,
    Signal,
    ResetSignal,
    traceSignals,
    now,
)
from dataclasses import dataclass, field
from itertools import tee, product
from typing import (
    Callable,
    Generator,
    Dict,
    List,
    Tuple,
    OrderedDict,
    Optional,
    Any,
    Union,
    Set,
)
from copy import deepcopy
from abc import ABC, abstractmethod, abstractproperty
import logging
from sys import version
import inspect
from functools import partial
from collections import Counter

# import showast
import ast
import astor
import re

logging.basicConfig(level=logging.INFO)
logging.info(version)
traceSignals.filename = "Top"
traceSignals.tracebackup = False


@dataclass
class StreamTemplate:
    _generator_func_def: Generator
    _stream_start_time_default = 0
    _stream_defualt_value_default = 0
    _stream_array_accessed_default = None

    def _parameterized_stream_descriptor(
        self,
        gen_func_args,
        gen_func_kwargs,
        local_stream_start_time,
        local_stream_defualt_value,
    ):
        customized_generator = self._generator_func_def(
            *gen_func_args, **gen_func_kwargs
        )
        for _ in range(local_stream_start_time):
            yield local_stream_defualt_value
        yield from customized_generator

    def __call__(self, *args, **kwargs):
        if "start_time" in kwargs:
            stream_start_time = kwargs["start_time"]
            del kwargs["start_time"]
        else:
            stream_start_time = self._stream_defualt_value_default

        if "array_accessed" in kwargs:
            stream_array_accessed = kwargs["array_accessed"]
            del kwargs["array_accessed"]
        else:
            stream_array_accessed = self._stream_array_accessed_default

        if "default_val" in kwargs:
            stream_defualt_value = kwargs["default_val"]
            del kwargs["default_val"]
        else:
            stream_defualt_value = self._stream_defualt_value_default

        return self._parameterized_stream_descriptor(
            args, kwargs, stream_start_time, stream_defualt_value
        )


def stream(stream_def_func):
    new_stream = StreamTemplate(_generator_func_def=stream_def_func)
    return new_stream


# Layer Config
ifmap_dim = 10
kernel = 3
ofmap_dim = ifmap_dim - kernel + 1
channel_count = 3

# Arch. Config For Full Channel Parallelism
pe_count = (kernel ** 2) * channel_count
pes_per_group = kernel
pes_per_channel = kernel ** 2
groups_per_channel = int(pes_per_channel / pes_per_group)
channel_chain_length = int(pe_count / pes_per_channel)


@stream
def example_func(c_ub, i_ub, j_ub, pe_channel, pe_group, pe, ifmap_dim):
    # Stream invariants
    pe_start_index_offset = pe_channel * (ifmap_dim ** 2) + pe_group * ifmap_dim + pe
    # Dynamic computations
    for c in range(c_ub, i_ub, 2):
        for i in range(i_ub):
            for j in range(j_ub):
                if 1 == 1 and j_ub == 2 and i == j and 4 == pe_start_index_offset:
                    if 4 == 3:
                        if 3 > 3 > 4:
                            yield i * ifmap_dim + j + pe_start_index_offset
                        else:
                            yield -1
                    else:
                        yield i * ifmap_dim + j + pe_start_index_offset

                elif 3 == 3:
                    yield i * ifmap_dim + j + pe_start_index_offset
                else:
                    yield i * ifmap_dim + j + pe_start_index_offset


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
                    raise SyntaxError(f"Duplicate iterator {iterator} in For loops")
        for val in vector:
            if val == "_":
                raise SyntaxError("Anonymous iterators not allowed in for loops")
        self._vector = vector


@dataclass
class AccessMap:
    access_expr: str = ""
    access_expr_with_annotated_parameters: str = ""
    condition: str = ""
    condition_with_annotated_parameters: str = ""
    parameters: Set[str] = field(default_factory=set)


@dataclass
class Invariant:
    name: str = ""


class NamedEntityAnnotator(ast.NodeVisitor):
    def __init__(self, _ignore) -> None:
        self.ignore = _ignore
        super().__init__()

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id not in self.ignore:
            node.id = f"{{{node.id}}}"
        self.generic_visit(node)

    @classmethod
    def annotate(cls, node, ignore=[]):
        _node = deepcopy(node)
        annotator = cls(ignore)
        annotator.visit(_node)
        return _node


class NamedEntityExtractor(ast.NodeVisitor):
    def __init__(self, _ignore) -> None:
        self.entity_set = set()
        self.ignore = _ignore
        super().__init__()

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id not in self.ignore:
            self.entity_set.add(node.id)
        self.generic_visit(node)

    @classmethod
    def extract(cls, node, ignore=[]):
        extractor = cls(ignore)
        extractor.visit(node)
        return extractor.entity_set


@dataclass
class StreamTokens:
    name: str = ""
    start_time: int = 0
    accessed_array: str = ""
    generator_args: ast.arguments = None
    yield_exprs: Tuple[Union[ast.Yield, Tuple[Union[ast.If, ast.Yield]]]] = ()
    invariant_assignments: Tuple[ast.Assign] = ()
    for_loops: Tuple[ast.For] = ()


@dataclass
class IslIR:
    name: str = ""
    iteration_domain: IterationDomain = IterationDomain()
    access_maps: Tuple[AccessMap] = ()
    invariants: Tuple[Invariant] = ()
    arguments: Dict[str, Union[int, None]] = field(default_factory=dict)


class ISLAbstractRepresentation:
    # TODO: Implement
    def __init__(self, ir: IslIR) -> None:
        #    self.iteration_domain = isl.BasicSet(iteration_domain.)
        pass


@dataclass
class ISLConcreteRepresentation:
    # TODO: Implement
    pass


class ISLGenerator:
    @staticmethod
    def get_string_repr(s):
        return s.__str__()

    @staticmethod
    def remove_brackets(s):
        return s[1:-1]

    @staticmethod
    def remove_single_quotes(s):
        return re.sub("'", "", s)

    @staticmethod
    def preprocess_iterable(it):
        return ISLGenerator.remove_single_quotes(
            ISLGenerator.remove_brackets(ISLGenerator.get_string_repr(it))
        )

    @classmethod
    def generate_abstract_repr(cls, ir: IslIR):
        params = ISLGenerator.preprocess_iterable(ir.iteration_domain.parameters)
        it_vector = ISLGenerator.preprocess_iterable(ir.iteration_domain.vector)

        bound_tuples = list(zip(ir.iteration_domain.vector, ir.iteration_domain.bounds))
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
        for idx, step in enumerate(ir.iteration_domain.steps):
            step_adjustment += "and"
            step_adjustment += f" {ir.iteration_domain.vector[idx]} mod {step} = 0 "

        structure = f"[{params}] -> {{ {ir.name.upper()}[{it_vector}] : {bound_expr} {step_adjustment}}}"

        print(isl.BasicSet(structure))

    @classmethod
    def generate_concrete_repr(ir):
        # TODO: Implement
        pass

    def check_access_map_aliasing(self):
        # TODO: Implement
        pass


class StreamParser:

    symbol_conversion_table = {r"==": "=", r"\n": "", r" +": " "}

    @classmethod
    def parse(cls, tokens: StreamTokens):
        parser = cls()
        parser.parse_name(tokens)
        parser.parse_arguments(tokens)
        parser.parse_invariants(tokens)
        parser.parse_iteration_domain(tokens)
        parser.parse_access_maps(tokens)
        return parser.ir

    def __init__(self):
        self.ir = IslIR()

    def convert_non_isl_symbols_to_isl_equivelent(self, expr_str):
        for symbol, target in StreamParser.symbol_conversion_table.items():
            expr_str = re.sub(symbol, target, expr_str)
        return expr_str

    def wrap_with_expr_and_fix_ast_entity(self, entity):
        wrapped_entity = ast.fix_missing_locations(ast.Expression(entity))
        return wrapped_entity

    def convert_expr_to_str(self, expr):
        expr_str = astor.to_source(expr)
        expr_str = self.convert_non_isl_symbols_to_isl_equivelent(expr_str)
        return expr_str

    def convert_access_map_condition_to_expr(self, condition, annotate_params=False):
        condition_expr = self.wrap_with_expr_and_fix_ast_entity(condition)
        if annotate_params:
            condition_expr = NamedEntityAnnotator.annotate(
                condition_expr, ignore=self.ir.iteration_domain.vector
            )
        return condition_expr

    def parse_name(self, tokens: StreamTokens):
        self.ir.name = tokens.name.upper()

    def parse_arguments(self, tokens: StreamTokens):
        if tokens.generator_args.vararg is not None:
            raise SyntaxError(f"Varargs not allowed in stream template {self.ir.name}")
        if tokens.generator_args.kwarg is not None:
            raise SyntaxError(f"Kwargs not allowed in stream template {self.ir.name}")

        num_of_args_with_default_val = len(tokens.generator_args.defaults)
        if num_of_args_with_default_val != 0:
            arg_list = tokens.generator_args.args[:(-num_of_args_with_default_val)]
        else:
            arg_list = tokens.generator_args.args

        for arg in arg_list:
            self.ir.arguments[arg.arg] = None
        for arg, default in zip(
            reversed(arg_list), reversed(tokens.generator_args.defaults)
        ):
            if not isinstance(default, ast.Constant) or not isinstance(
                default.value, int
            ):
                raise SyntaxError(
                    f"Stream template arg: {arg.arg} has a non-int default: {default.value}"
                )
            self.ir.arguments[arg.arg] = default

    def parse_invariants(self, tokens: StreamTokens):
        invariant_targets = [
            assignment.targets[0].id for assignment in tokens.invariant_assignments
        ]
        for idx_current_assignment, assignment in enumerate(
            tokens.invariant_assignments
        ):
            target = assignment.targets[0].id
            if len(assignment.targets) > 1:
                raise SyntaxError(
                    f"Invariant assignment with target '{target}' can only have one target"
                )
            assignment_vars = NamedEntityExtractor.extract(assignment.value)
            for var in assignment_vars:
                if var not in invariant_targets and var not in self.ir.arguments.keys():
                    raise SyntaxError(
                        f"Invalid variable in invariant assignment target '{target}', '{var}' is not a stream template arg nor is it another invariant"
                    )
                if var in invariant_targets:
                    idx_of_var_in_targets = invariant_targets.index(var)
                    if idx_of_var_in_targets >= idx_current_assignment:
                        raise SyntaxError(
                            f"Invariant assignment '{target}' references '{var}' before its assignment"
                        )
        self.ir.invariants = tuple(invariant_targets)

    def parse_access_maps(self, tokens: StreamTokens):
        for expr in tokens.yield_exprs:
            condition_list = expr[:-1]
            yield_expr = expr[-1]

            # Get parameter list and validate
            chain_parameter_set = set()
            for condition in condition_list:
                for param in NamedEntityExtractor.extract(
                    condition.test, ignore=self.ir.iteration_domain.vector
                ):
                    if (
                        param not in self.ir.arguments
                        and param not in self.ir.invariants
                    ):
                        raise SyntaxError(
                            f"Parameter {param} in expression \n'{astor.to_source(condition.test).strip()}'\nis not a stream argument nor an invariant"
                        )
                    chain_parameter_set.add(param)

            for param in NamedEntityExtractor.extract(
                yield_expr, ignore=self.ir.iteration_domain.vector
            ):
                if param not in self.ir.arguments and param not in self.ir.invariants:
                    raise SyntaxError(
                        f"Parameter {param} in expression \n'{astor.to_source(yield_expr.value).strip()}'\nis not a stream argument nor an invariant"
                    )
                chain_parameter_set.add(param)

            # Get condition
            chain_conditions_expr = self.convert_expr_to_str(
                ast.BoolOp(
                    op=ast.And(),
                    values=[
                        self.convert_access_map_condition_to_expr(condition.test)
                        for condition in condition_list
                    ],
                )
            )

            # Get annotated condition
            chain_conditions_expr_with_annotated_params = self.convert_expr_to_str(
                ast.BoolOp(
                    op=ast.And(),
                    values=[
                        self.convert_access_map_condition_to_expr(
                            condition.test, annotate_params=True
                        )
                        for condition in condition_list
                    ],
                )
            )

            # Get access expression
            yield_expr_with_annotated_params = self.convert_expr_to_str(
                NamedEntityAnnotator.annotate(
                    yield_expr.value, ignore=self.ir.iteration_domain.vector
                )
            )
            yield_expr = self.convert_expr_to_str(yield_expr.value)

            self.ir.access_maps += (
                AccessMap(
                    access_expr=yield_expr,
                    access_expr_with_annotated_parameters=yield_expr_with_annotated_params,
                    condition=chain_conditions_expr,
                    condition_with_annotated_parameters=chain_conditions_expr_with_annotated_params,
                    parameters=chain_parameter_set,
                ),
            )

    def parse_iteration_domain(self, tokens: StreamTokens):
        # Get iteration domain vectors
        self.ir.iteration_domain.vector = tuple(
            [loop.target.id for loop in tokens.for_loops]
        )

        for loop in tokens.for_loops:
            if isinstance(loop.iter, ast.Call) and loop.iter.func.id == "range":
                for arg in loop.iter.args:
                    if isinstance(arg, ast.Name):
                        if (
                            arg.id not in self.ir.arguments
                            and arg not in self.ir.invariants
                        ):
                            raise SyntaxError(
                                f"Argument '{arg.id}' in loop \n'{astor.to_source(loop).strip()}'\nis not a stream argument nor an invariant"
                            )
                        self.ir.iteration_domain.parameters.add(arg.id)
                    # TODO: Handle non name instance e.g. function
                bounds = ()
                for arg in loop.iter.args[:2]:
                    try:
                        bounds += (arg.value,)
                    except AttributeError:
                        bounds += (arg.id,)
                    except AttributeError:
                        raise SyntaxError(
                            f"Invalid bound argument '{arg}' for loop range"
                        )
                self.ir.iteration_domain.bounds += (bounds,)
                if len(loop.iter.args) == 3:
                    arg = loop.iter.args[2]
                    try:
                        self.ir.iteration_domain.steps += (arg.value,)
                    except AttributeError:
                        raise SyntaxError(
                            f"Invalid step argument '{arg.id}' in for loop range, only constants are allowed"
                        )
            else:
                raise SyntaxError(
                    f"For loops with iterator '{loop.target.id}' can only iterate over ranges"
                )


class StreamExtractor(astor.ExplicitNodeVisitor):
    @classmethod
    def extract(cls, node, start_time=None, accessed_array=None):
        extractor = cls()
        extractor.visit(node)
        extractor.prune_orelse()
        return extractor.tokens

    def __init__(self):
        self.tokens = StreamTokens()

    def prune_orelse(self):
        for yield_expr in self.tokens.yield_exprs:
            for condition in yield_expr:
                if isinstance(condition, ast.If):
                    condition.orelse = []
                elif isinstance(condition, ast.Yield):
                    break
                else:
                    raise Exception(
                        "Found non condition or yield node when pruning yield expressions orelse conditions"
                    )

    def visit_Module(self, node: ast.Module) -> Any:
        stream_func_def_node = node.body[0]
        self.visit_FunctionDef(stream_func_def_node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.tokens.name = node.name
        if node.decorator_list[0].id != "stream":
            raise SyntaxError(
                f"Invalid function '{node.name}' used for conversion to ISL IR"
            )
        self.tokens.generator_args = node.args
        start_of_for_loops_idx = 0
        for idx, _node in enumerate(node.body):
            if isinstance(_node, ast.Assign):
                self.tokens.invariant_assignments += (_node,)
            elif isinstance(_node, ast.For):
                start_of_for_loops_idx = idx
                break
            else:
                raise SyntaxError(
                    f"Invalid expression '{astor.to_source(_node).strip()}' in '{node.name}' body. Only constant assignments, and for loops are allowed"
                )

        if start_of_for_loops_idx != len(node.body) - 1:
            logging.warning(
                "Other expressions in function definition beyond the first for loop will be ignored"
            )
        self.visit_For(node.body[start_of_for_loops_idx])

    def visit_For(self, node: ast.For) -> Any:
        self.tokens.for_loops += (node,)
        if len(node.body) > 1:
            logging.warning(
                f"Multiple statements in body of for loop with iterator '{node.target.id}', only the first statement will be parsed"
            )
        first_entry_in_loop_body = node.body[0]
        if isinstance(first_entry_in_loop_body, ast.For):
            self.visit_For(first_entry_in_loop_body)
        elif isinstance(first_entry_in_loop_body, ast.If):
            self.visit_If(first_entry_in_loop_body)
        elif isinstance(first_entry_in_loop_body, ast.Expr):
            if isinstance(first_entry_in_loop_body.value, ast.Yield):
                self.visit_Yield(first_entry_in_loop_body.value)
            else:
                raise SyntaxError(
                    f"Invalid expression '{astor.to_source(first_entry_in_loop_body).strip()}' in for loop body, only yield expressions are allowed"
                )
        else:
            raise SyntaxError(
                f"Invalid statement '{astor.to_source(first_entry_in_loop_body).strip()}' in for loop with iterator  '{node.target.id}' body, only other for loops, if statements, and yield expressions are allowed"
            )

    def visit_Yield(self, node: ast.Yield, if_chain: Tuple = ()) -> Any:
        if_chain += (deepcopy(node),)
        self.tokens.yield_exprs += deepcopy((if_chain,))

    def visit_If(self, node: ast.If, if_chain=()) -> Any:
        if_chain += (deepcopy(node),)
        if len(node.body) > 1:
            raise SyntaxError(
                f"Invalid additional expression in if/elif/else body \n'{astor.to_source(node.body[1]).strip()}'\n\nonly one expression is allowed in if/elif/else body"
            )
        if len(node.orelse) > 1:
            raise SyntaxError(
                f"Invalid additional expression in if/elif/else body \n'{astor.to_source(node.orelse[1]).strip()}'\n\nonly one expression is allowed in if/elif/else body"
            )

        if len(node.body) == 1:
            entry = node.body[0]
            if isinstance(entry, ast.If):
                self.visit_If(entry, if_chain)
            elif isinstance(entry, ast.Expr):
                if isinstance(entry.value, ast.Yield):
                    self.visit_Yield(entry.value, if_chain)
                else:
                    raise SyntaxError(
                        f"Invalid statement \n'{astor.to_source(entry)}'\n in if condition body, only yield expression are allowed"
                    )
            else:
                raise SyntaxError(
                    f"Invalid statement \n'{astor.to_source(entry)}'\n , only other if statements, and yield expressions are allowed"
                )
        if len(node.orelse) == 1:
            entry = node.orelse[0]
            if_chain[-1].test = ast.UnaryOp(op=ast.Not(), operand=if_chain[-1].test)
            if isinstance(entry, ast.If):
                self.visit_If(entry, if_chain)
            elif isinstance(entry, ast.Expr):
                if isinstance(entry.value, ast.Yield):
                    self.visit_Yield(entry.value, if_chain)
                else:
                    raise SyntaxError(
                        f"Invalid statement \n'{astor.to_source(entry)}'\n in if condition body, only yield expression are allowed"
                    )
            else:
                raise SyntaxError(
                    f"Invalid statement \n'{astor.to_source(entry)}'\n in if condition body, only other if statements, and yield expressions are allowed"
                )


tree = ast.parse(
    inspect.getsource(
        inspect.getgeneratorlocals(
            example_func(1, ofmap_dim, ofmap_dim, 0, 0, 0, ifmap_dim, start_time=0)
        )["self"]._generator_func_def
    )
)


tokens = StreamExtractor.extract(tree)
ir = StreamParser.parse(tokens)
abs_repr = ISLGenerator.generate_abstract_repr(ir)
print("DONE")
