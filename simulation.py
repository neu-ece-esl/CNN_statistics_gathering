# To add a new cell, type '
# To add a new markdown cell, type  [markdown]'

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
)
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
traceSignals.filename = "Top"
traceSignals.tracebackup = False


@dataclass
class StreamStateControl:
    index_generator_fn: Generator
    initial_index_generator_fn: Generator = None
    _done: bool = False

    def __post_init__(self):
        self.index_generator_fn, self.initial_index_generator_fn = tee(
            self.index_generator_fn
        )

    def reset(self):
        self.done = False
        self.index_generator_fn = self.initial_index_generator_fn
        self.index_generator_fn, self.initial_index_generator_fn = tee(
            self.index_generator_fn
        )

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


@block
def counter(clk, enable, reset, count):
    @always_seq(clk.posedge, reset=reset)
    def increment():
        if enable:
            count.next = count.val + 1

    return increment


@block
def clk_driver(clk, enable, period=20):
    lowTime = int(period / 2)
    highTime = period - lowTime

    @instance
    def drive_clk():
        while True:
            if not enable:
                yield enable
            yield delay(lowTime)
            clk.next = 1
            yield delay(highTime)
            clk.next = 0

    return drive_clk


@block
def stream_generator(clk, enable, reset, stream, stream_out):
    @always(clk.posedge, reset.posedge)
    def generate():
        if not reset and enable:
            if not stream.done:
                stream_out.next = next(stream)
        elif reset:
            stream.reset()
            stream_out.next = 0

    return generate


@stream
def chain_arch_pe_parameterizable_access_stream(
    c_ub, i_ub, j_ub, pe_channel, pe_group, pe, ifmap_dim
):
    # Stream constants
    pe_start_index_offset = pe_channel * (ifmap_dim ** 2) + pe_group * ifmap_dim + pe
    print(pe_start_index_offset)

    # Dynamic computations
    for c in range(c_ub):
        for i in range(i_ub):
            for j in range(j_ub):
                yield i * ifmap_dim + j + pe_start_index_offset


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


@block
def top():
    clk = Signal(bool(0))
    enable = Signal(bool(0))
    global_counter = Signal(0)
    reset = ResetSignal(bool(0), active=1, isasync=True)
    counter_inst = counter(clk, enable, reset, global_counter)
    clk_driver_inst = clk_driver(clk, enable, period=10)

    stream_out_list = [Signal(0) for _ in range(pe_count)]

    stream_generator_list = []
    for pe_channel in range(channel_chain_length):
        for pe_group in range(groups_per_channel):
            for pe in range(pes_per_group):
                pe_idx = pe_channel * pes_per_channel + pe_group * pes_per_group + pe
                stream_descriptor = chain_arch_pe_parameterizable_access_stream(
                    1,
                    ofmap_dim,
                    ofmap_dim,
                    pe_channel,
                    pe_group,
                    pe,
                    ifmap_dim,
                    start_time=pe_idx + 1,
                    default_val=-1,
                )
                stream_state_controller = StreamStateControl(stream_descriptor)
                stream_generator_list.append(
                    stream_generator(
                        clk,
                        enable,
                        reset,
                        stream_state_controller,
                        stream_out_list[pe_idx],
                    )
                )

    @instance
    def start_sim():
        # reset cycle
        enable.next = 0
        reset.next = 1
        yield delay(10)
        enable.next = 1
        reset.next = 0

    return clk_driver_inst, counter_inst, start_sim, stream_generator_list


dut = top()
inst = traceSignals(dut)
inst.run_sim(1200)
inst.quit_sim()


# utilization = ((pe_count+1)/2+((ofmap_dim**2)-(pe_count-1))+(pe_count+1)/2)/((pe_count-1)+ofmap_dim**2)
# print(utilization)


# tree = ast.parse(inspect.getsource(inspect.getgeneratorlocals(example_func(
#                     1, ofmap_dim, ofmap_dim, 0, 0, 0, ifmap_dim, start_time=0))['self']._generator_func_def))
