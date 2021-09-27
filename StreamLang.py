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
