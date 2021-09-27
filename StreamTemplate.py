from dataclasses import dataclass
from typing import (
    Generator,
)

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