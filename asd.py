# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

from ast import Eq
import islpy as isl
from myhdl import block, delay, always_seq, instance, always, Signal, ResetSignal, traceSignals, now
from dataclasses import dataclass, field
from itertools import tee, product
from typing import Callable, Generator, Dict, List, Tuple, OrderedDict, Optional, Any, Union
from copy import deepcopy
from abc import ABC, abstractmethod, abstractproperty
import logging
from sys import version
import inspect
from functools import partial
# import showast
import ast
import astor
logging.basicConfig(level=logging.INFO)
logging.info(version)
traceSignals.filename = 'Top'
traceSignals.tracebackup = False


@dataclass
class StreamStateControl:
    index_generator_fn: Generator
    initial_index_generator_fn: Generator = None
    _done: bool = False

    def __post_init__(self):
        self.index_generator_fn, self.initial_index_generator_fn = tee(
            self.index_generator_fn)

    def reset(self):
        self.done = False
        self.index_generator_fn = self.initial_index_generator_fn
        self.index_generator_fn, self.initial_index_generator_fn = tee(
            self.index_generator_fn)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            next_index = next(self.index_generator_fn)
        except StopIteration:
            next_index = 0
            self.done = True
        return next_index

    @property
    def done(self):
        return self._done

    @done.setter
    def done(self, val):
        if val:
            logging.debug("{} has concluded @T={}".format(self, now()))
        else:
            logging.debug("{} initialized @T={}".format(self, now()))
        self._done = val


@dataclass
class StreamTemplate:
    _generator_func_def: Generator
    _stream_start_time_default = 0
    _stream_defualt_value_default = 0
    _stream_array_accessed_default = None

    def _parameterized_stream_descriptor(self, gen_func_args, gen_func_kwargs, local_stream_start_time, local_stream_defualt_value):
        customized_generator = self._generator_func_def(
            *gen_func_args, **gen_func_kwargs)
        for _ in range(local_stream_start_time):
            yield local_stream_defualt_value
        yield from customized_generator

    def __call__(self, *args, **kwargs):
        if 'start_time' in kwargs:
            stream_start_time = kwargs['start_time']
            del kwargs['start_time']
        else:
            stream_start_time = self._stream_defualt_value_default

        if 'array_accessed' in kwargs:
            stream_array_accessed = kwargs['array_accessed']
            del kwargs['array_accessed']
        else:
            stream_array_accessed = self._stream_array_accessed_default

        if 'default_val' in kwargs:
            stream_defualt_value = kwargs['default_val']
            del kwargs['default_val']
        else:
            stream_defualt_value = self._stream_defualt_value_default

        return self._parameterized_stream_descriptor(args, kwargs, stream_start_time, stream_defualt_value)


def stream(stream_def_func):
    new_stream = StreamTemplate(_generator_func_def=stream_def_func)
    return new_stream


# Layer Config
ifmap_dim = 10
kernel = 3
ofmap_dim = ifmap_dim-kernel+1
channel_count = 3

# Arch. Config For Full Channel Parallelism
pe_count = (kernel**2)*channel_count
pes_per_group = kernel
pes_per_channel = kernel**2
groups_per_channel = int(pes_per_channel/pes_per_group)
channel_chain_length = int(pe_count/pes_per_channel)


# %%
@stream
def example_func(c_ub, i_ub, j_ub, pe_channel, pe_group, pe, ifmap_dim, test=1, *args, **kwargs):
    # Stream invariants
    pe_start_index_offset = pe_channel*(ifmap_dim**2)+pe_group*ifmap_dim+pe
    # Dynamic computations
    for c in range(c_ub, i_ub, j_ub):
        for i in range(i_ub):
            for j in range(j_ub):
                if 1 == 1 and j_ub == 2 and asd == dsa and 4 == ass:
                    if 4 == 3:
                        if 3 > a > 4:
                            yield i*ifmap_dim+j+pe_start_index_offset
                        else:
                            yield 3
                    else:
                        yield i*ifmap_dim+j+pe_start_index_offset

                elif 3 == 3:
                    yield i*ifmap_dim+j+pe_start_index_offset
                else:
                    yield i*ifmap_dim+j+pe_start_index_offset


@dataclass
class IterationDomain:
    vector: Tuple[str] = ()
    bounds: Tuple[Tuple[Union[int, str]]] = (())
    steps: Tuple[Tuple[Union[int, str]]] = ()
    parameters: Tuple[str] = ()


@dataclass
class AccessMap:
    access_expr: str = ''
    access_expr_with_annotated_parameters: str = ''
    condition: str = ''
    condition_with_annotated_parameters: str = ''
    parameters: Tuple[str] = ()

@dataclass
class Invariant:
    name: str = ''

class NamedEntityAnnotator(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()

    def visit_Name(self, node: ast.Name) -> Any:
        node.id = f'{{{node.id}}}'
        self.generic_visit(node)

    @classmethod
    def annotate(cls, node):
        _node = deepcopy(node)
        annotator = cls()
        annotator.visit(_node)
        return _node


class NamedEntityExtractor(ast.NodeVisitor):

    def __init__(self) -> None:
        self.entity_set = set()
        super().__init__()

    def visit_Name(self, node: ast.Name) -> Any:
        self.entity_set.add(node.id)
        self.generic_visit(node)

    @classmethod
    def extract(cls, node):
        extractor = cls()
        extractor.visit(node)
        return extractor.entity_set


@dataclass
class StreamTokens:
    name: str = ''
    args: ast.arguments = None
    yield_exprs: Tuple[Union[ast.Yield, Tuple[Union[ast.If, ast.Yield]]]] = ()
    stream_invariant_assignments: Tuple[ast.Assign] = ()
    for_loops: Tuple[ast.For] = ()

@dataclass
class IslIR:
    iteration_domain: IterationDomain = IterationDomain()
    access_maps: Tuple[AccessMap] = ()
    invariants : Tuple[Invariant] = ()
    
    def convert_non_isl_symbols_to_isl_equivelent(self, condition):
        return condition

    def wrap_ast_entity_with_expr(self, entity):
        wrapped_entity = ast.fix_missing_locations(
            ast.Expression(entity))
        return wrapped_entity

    def convert_expr_to_str(self, expr):
        return astor.to_source(expr)

    def convert_access_map_condition_to_expr(self, condition, annotate_params=False):
        condition_expr = self.wrap_ast_entity_with_expr(condition)
        if annotate_params:
            condition_expr = NamedEntityAnnotator.annotate(condition_expr)
        return condition_expr

    def parse_name(self, tokens : StreamTokens):
        pass

    def parse_arguments(self, tokens : StreamTokens):
        pass

    def parse_invariants(self, tokens : StreamTokens):
        pass

    def parse_access_maps(self, tokens : StreamTokens):
        for expr in self.yield_expr:
            condition_list = expr[:-1]
            yield_expr = expr[-1]

            # Get parameter list
            chain_parameter_set = set()
            for condition in condition_list:
                for param in NamedEntityExtractor.extract(condition.test):
                    chain_parameter_set.add(param)
            for param in NamedEntityExtractor.extract(yield_expr):
                chain_parameter_set.add(param)

            chain_parameter_set = set(
                [param for param in chain_parameter_set if param not in self.iteration_domain.vector])
            # Get condition
            chain_conditions_expr = self.convert_expr_to_str(
                ast.BoolOp(op=ast.And(), values=[
                    self.convert_access_map_condition_to_expr(condition.test)
                    for condition in condition_list
                ]))

            # Get annotated condition
            chain_conditions_expr_with_annotated_params = self.convert_expr_to_str(
                ast.BoolOp(op=ast.And(), values=[
                    self.convert_access_map_condition_to_expr(
                        condition.test, annotate_params=True)
                    for condition in condition_list
                ]))

            # Get access expression
            yield_expr_with_annotated_params = self.convert_expr_to_str(
                NamedEntityAnnotator.annotate(yield_expr.value))
            yield_expr = self.convert_expr_to_str(yield_expr.value)

            self.access_maps += (
                AccessMap(access_expr=yield_expr,
                          access_expr_with_annotated_parameters=yield_expr_with_annotated_params,
                          condition=chain_conditions_expr,
                          condition_with_annotated_parameters=chain_conditions_expr_with_annotated_params,
                          parameters=chain_parameter_set),
            )

    def parse_iteration_domain(self, tokens : StreamTokens):
        self.iteration_domain.vector = tuple(
            [loop.target.id for loop in self.for_loops])
        for loop in self.for_loops:
            if isinstance(loop.iter, ast.Call) and loop.iter.func.id == 'range':
                self.iteration_domain.parameters += tuple(
                    [arg.id for arg in loop.iter.args if isinstance(arg, ast.Name)])
                bounds = ()
                for arg in loop.iter.args[:2]:
                    try:
                        bounds += (arg.value, )
                    except AttributeError:
                        bounds += (arg.id, )
                    except AttributeError:
                        raise Exception(
                            "Invalid bound argument(s) for loop range")
                self.iteration_domain.bounds += (bounds, )
                if len(loop.iter.args) == 3:
                    arg = loop.iter.args[2]
                    try:
                        self.iteration_domain.steps += (arg.value, )
                    except AttributeError:
                        self.iteration_domain.steps += (arg.id, )
                    except AttributeError:
                        raise Exception(
                            "Invalid bound argument(s) for loop range")
            else:
                raise Exception("For loops can only iterate over ranges")
        self.iteration_domain.parameters = tuple(
            set(self.iteration_domain.parameters))


class StreamParser(astor.ExplicitNodeVisitor):

    @classmethod
    def parse(cls, node):
        parser = cls()
        parser.visit(node)
        parser.prune_orelse()
        return parser.tokens

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
                        "Found non condition or yield node when pruning yield expressions orelse conditions")

    def visit_Module(self, node: ast.Module) -> Any:
        stream_func_def_node = node.body[0]
        self.visit_FunctionDef(stream_func_def_node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        if node.decorator_list[0].id != 'stream':
            raise Exception("Invalid function used for conversion to ISL IR")
        self.tokens.name = node.name
        self.tokens.stream_args = node.args
        start_of_for_loops_idx = 0
        for idx, _node in enumerate(node.body):
            if isinstance(_node, ast.Assign):
                self.tokens.stream_invariant_assignments += (_node,)
            elif isinstance(_node, ast.For):
                start_of_for_loops_idx = idx
                break
            else:
                raise Exception(
                    "Invalid expression in function body, only constant assignments, and for loops are allowed")

        if start_of_for_loops_idx != len(node.body) - 1:
            logging.warning(
                "Other expressions in function definition beyond the first for loop will be ignored")
        self.visit_For(node.body[start_of_for_loops_idx])

    def visit_For(self, node: ast.For) -> Any:
        self.tokens.for_loops += (node, )
        first_entry_in_loop_body = node.body[0]
        if isinstance(first_entry_in_loop_body, ast.For):
            self.visit_For(first_entry_in_loop_body)
        elif isinstance(first_entry_in_loop_body, ast.If):
            self.visit_If(first_entry_in_loop_body)
        elif isinstance(first_entry_in_loop_body, ast.Expr):
            if isinstance(first_entry_in_loop_body.value, ast.Yield):
                self.visit_Yield(first_entry_in_loop_body.value)
            else:
                raise Exception(
                    "Invalid expression type in for loop body, only yield expressions are allowed")
        else:
            raise Exception(
                "Invalid for body, only for loops, if statements, and yield expressions are allowed in a for loop")

    def visit_Yield(self, node: ast.Yield, if_chain: Tuple = ()) -> Any:
        if_chain += (deepcopy(node), )
        self.tokens.yield_exprs += deepcopy((if_chain, ))

    def visit_If(self, node: ast.If, if_chain=()) -> Any:
        if_chain += (deepcopy(node), )
        if len(node.body) > 1 or len(node.orelse) > 1:
            raise Exception(
                "Too many expressions in if/elif/else statement bodies, something fishy is going on....")
        if len(node.body) == 1:
            entry = node.body[0]
            if isinstance(entry, ast.If):
                self.visit_If(entry, if_chain)
            elif isinstance(entry, ast.Expr):
                if(isinstance(entry.value, ast.Yield)):
                    self.visit_Yield(entry.value, if_chain)
                else:
                    raise Exception(
                        "Invalid expression type in if condition body, only yield expression are allowed")
            else:
                raise Exception(
                    "Invalid if condition body, only other if statements, and yield expressions are allowed")
        if len(node.orelse) == 1:
            entry = node.orelse[0]
            if_chain[-1].test = ast.UnaryOp(op=ast.Not(),
                                            operand=if_chain[-1].test)
            if isinstance(entry, ast.If):
                self.visit_If(entry, if_chain)
            elif isinstance(entry, ast.Expr):
                if(isinstance(entry.value, ast.Yield)):
                    self.visit_Yield(entry.value, if_chain)
                else:
                    raise Exception(
                        "Invalid expression type in if condition body, only yield expression are allowed")
            else:
                raise Exception(
                    "Invalid if condition body, only other if statements, and yield expressions are allowed")

tree = ast.parse(inspect.getsource(inspect.getgeneratorlocals(example_func(
    1, ofmap_dim, ofmap_dim, 0, 0, 0, ifmap_dim, start_time=0))['self']._generator_func_def))


# %%
tokens = StreamParser.parse(tree)
# parser.isl_ir.parse_iteration_domain()
# parser.isl_ir.parse_access_maps()
print("DONE")