import torch
from collections import OrderedDict, Counter
from functools import partial
from dataclasses import dataclass
from typing import Tuple, List
from copy import deepcopy
from math import prod


class StatsCounter:
    def __init__(self):
        self._dict = {}
    def update(self, key, v = None):
        if key not in self._dict:
            self._dict[key] = 1
        else:
            self._dict[key] += v if v is not None else 1
    def __iadd__(self, other):
        if isinstance(other, StatsCounter):
            for k, v in other.items():
                self.update(k, v)
        else:
            raise TypeError("Can only add other StatsCounters to other StatsCounters")
        return self
    def __getitem__(self, key):
        return self._dict[key]
    def __str__(self):
        return self._dict.__str__()
    def __repr__(self):
        return self._dict.__repr__() 
    def items(self):
        return self._dict.items()

@dataclass
class LayerDimensions:
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    input_size: List[int]
    output_size: List[int]


class ModelStatCollector:
    def __init__(self):
        self.model_stats = OrderedDict()
        self.hooks = []
        
    def __get_next_conv_layers(self, model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                yield (name, module)

    def __extract_stats(self, name, module, input, output):
        self.model_stats[name] = LayerDimensions(module.kernel_size, module.stride, module.padding, input_size=list(
            input[0].size()), output_size=list(output[0].size()))

    def __attach_collection_hooks_to_model(self, model):
        for name, conv_layer in self.__get_next_conv_layers(model):
            layer_collector = partial(self.__extract_stats, name)
            self.hooks.append(
                conv_layer.register_forward_hook(layer_collector))

    def __detach_stats_collection_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __reset(self):
        self.model_stats = {}
        self.hooks = []

    @ classmethod 
    def collect_stats_from_model(cls, model, input_batch):
        collector = cls()
        collector.__attach_collection_hooks_to_model(model)
        model.eval()
        with torch.no_grad():
            model(input_batch)
        collector.__detach_stats_collection_hooks()
        collected_stats = deepcopy(collector.model_stats)
        collector.__reset()
        return collected_stats

class ModelStatAnalyser:
    @ classmethod
    def get_kernel_stats(cls, model_stats):
        kernel_size_counter = StatsCounter()
        stride_counter_dict = {}
        for layer in model_stats.values():
            kernel_size = layer.kernel_size
            stride = layer.stride
            kernel_size_counter.update(f'{kernel_size[0]}')
            if f'{kernel_size[0]}' not in stride_counter_dict:
                stride_counter_dict[f'{kernel_size[0]}'] = StatsCounter()
            stride_counter_dict[f'{kernel_size[0]}'].update(f'{stride[0]}')
        return kernel_size_counter

    @ classmethod
    def get_intermediate_layer_sizes(cls, model_stats):
        intermediate_layer_sizes = [
            (prod(layer.input_size), layer.input_size) for layer in model_stats.values()]
        intermediate_layer_sizes.append(
            (prod(list(model_stats.values())[-1].output_size), list(model_stats.values())[-1].output_size))
        return intermediate_layer_sizes

    @ classmethod
    def get_intermediate_layer_size_bounds(cls, model_stats):
        intermediate_layer_sizes = cls.get_intermediate_layer_sizes(model_stats)
        max_layer_size = max(intermediate_layer_sizes, key=lambda elem: elem[0])
        min_layer_size = min(intermediate_layer_sizes, key=lambda elem: elem[0])
        return (max_layer_size,
                min_layer_size)

    @ classmethod
    def get_ub_input_size(cls, model_stats):
        return max([prod(layer.kernel_size[0]) for layer in model_stats.values()])

    @ classmethod
    def get_in_channel_stats(cls, model_stats):
        in_channel_dict = {}
        for layer in model_stats.values():
            kernel_size = layer.kernel_size
            if f'{kernel_size[0]}' not in in_channel_dict:
                in_channel_dict[f'{kernel_size[0]}'] = {}
            if f'{layer.input_size[1]}' not in in_channel_dict[f'{kernel_size[0]}']:
                in_channel_dict[f'{kernel_size[0]}'][f'{layer.input_size[1]}'] = 0
            in_channel_dict[f'{kernel_size[0]}'][f'{layer.input_size[1]}'] += 1

        return in_channel_dict

    @ classmethod
    def get_filter_stats(cls, model_stats):
        out_channel_dict = {}
        for layer in model_stats.values():
            kernel_size = layer.kernel_size
            if f'{kernel_size[0]}' not in out_channel_dict:
                out_channel_dict[f'{kernel_size[0]}'] = {}
            if f'{layer.output_size[0]}' not in out_channel_dict[f'{kernel_size[0]}']:
                out_channel_dict[f'{kernel_size[0]}'][f'{layer.output_size[0]}'] = 0
            out_channel_dict[f'{kernel_size[0]}'][f'{layer.output_size[0]}'] += 1

        return out_channel_dict

    @ classmethod
    def get_stride_stats(cls, model_stats):
        stride_counter_dict = {}
        for layer in model_stats.values():
            kernel_size = layer.kernel_size
            stride = layer.stride
            if f'{kernel_size[0]}' not in stride_counter_dict:
                stride_counter_dict[f'{kernel_size[0]}'] = StatsCounter()
            stride_counter_dict[f'{kernel_size[0]}'].update(f'{stride[0]}')
        return stride_counter_dict

    @ classmethod 
    def get_models_stats_dict(cls, model_dict, input_batch, ssd_input_batch = None):
        stats_dict = {}
        raw_stats_dict = {}
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            if ssd_input_batch is not None:
                ssd_input_batch = ssd_input_batch.to('cuda')
        for model_name, model in model_dict.items():
            model.to('cuda')
            if model_name == 'ssd' and ssd_input_batch is not None:
                model_stats = ModelStatCollector.collect_stats_from_model(model, ssd_input_batch)
            else:
                model_stats = ModelStatCollector.collect_stats_from_model(model, input_batch)
                
            model.to('cpu')
            stats_dict[model_name] = {'kernel': ModelStatAnalyser.get_kernel_stats(model_stats),
                                    'stride': ModelStatAnalyser.get_stride_stats(model_stats),
                                    'in_channel': ModelStatAnalyser.get_in_channel_stats(model_stats),
                                    'filters': ModelStatAnalyser.get_filter_stats(model_stats),
                                    'intermediate_layer_sizes': ModelStatAnalyser.get_intermediate_layer_sizes(model_stats),
                                    'intermediate_layer_bounds': ModelStatAnalyser.get_intermediate_layer_size_bounds(model_stats),
                                    'raw_stats': model_stats}
        return stats_dict
    
class ModelStatsAggregator:
    
    @ classmethod
    def get_aggregate_kernel_stats_as_percentages(cls, stats_dict):
        aggregate_kernel_stats = StatsCounter()
        for model, stats in stats_dict.items():
            aggregate_kernel_stats += stats['kernel']
        ksize, counts = zip(*[(ksize, count) for ksize, count in aggregate_kernel_stats.items()])
        total_kernels = sum(counts)
        aggregate_kernel_stats_percentages = {ksize: counts/total_kernels for ksize, counts in zip(ksize,counts)}
        return aggregate_kernel_stats_percentages

    @ classmethod
    def get_aggregate_stride_stats_per_kernel(cls, stats_dict):
        aggregate_stride_stats = {}
        for model, stats in stats_dict.items():
            for kernel, counter in stats['stride'].items():
                if kernel not in aggregate_stride_stats:
                    aggregate_stride_stats[kernel] = StatsCounter()
                aggregate_stride_stats[kernel] += counter
        return aggregate_stride_stats
    
    @ classmethod
    def get_stride_stats_per_kernel_as_percentages(cls, stats_dict):
        aggregate_stride_stats = cls.get_aggregate_stride_stats_per_kernel(stats_dict)
        aggregate_stride_stats_percentages = {}
        for ksize, stride_counter in aggregate_stride_stats.items():
            total_kernels = sum(stride_counter.values())
            aggregate_stride_stats_percentages[ksize] = {stride: count/total_kernels for stride, count in dict(stride_counter).items()}
        return aggregate_stride_stats_percentages
    
    @ classmethod
    def get_aggregate_in_channel_stats_per_kernel(cls, stats_dict):
        aggregate_in_channel_stats = {}
        for _, stats in stats_dict.items():
            for kernel, channel_dict in stats['in_channel'].items():
                if kernel not in aggregate_in_channel_stats:
                    aggregate_in_channel_stats[kernel] = {}
                for channel, count in channel_dict.items():
                    if channel not in aggregate_in_channel_stats[kernel]:
                        aggregate_in_channel_stats[kernel][channel] = 0
                    aggregate_in_channel_stats[kernel][channel] += count
        for kernel, channel_dict in aggregate_in_channel_stats.items():
            channel_dict = {k: v for k, v in sorted(channel_dict.items(), key=lambda item: item[1], reverse=True)}
            aggregate_in_channel_stats[kernel] = channel_dict
        return aggregate_in_channel_stats
    
    @ classmethod
    def get_aggregate_in_channel_stats(cls, stats_dict):
        aggregate_in_channel_stats = cls.get_aggregate_in_channel_stats_per_kernel(stats_dict)
        cross_kernel_aggregate_channel = {}
        for channel_dicts in aggregate_in_channel_stats.values():
            for channel_size, count in channel_dicts.items():
                if channel_size not in cross_kernel_aggregate_channel:
                    cross_kernel_aggregate_channel[channel_size] = 0
                cross_kernel_aggregate_channel[channel_size] += count
        cross_kernel_aggregate_channel = {k: v for k,v in sorted(cross_kernel_aggregate_channel.items(), key=lambda item: item[1], reverse=True)}
        return cross_kernel_aggregate_channel
        
    @ classmethod
    def get_aggregate_filter_stats_per_kernel(cls, stats_dict):
        aggregate_filter_stats = {}
        for model, stats in stats_dict.items():
            for kernel, channel_dict in stats['filters'].items():
                if kernel not in aggregate_filter_stats:
                    aggregate_filter_stats[kernel] = {}
                for channel, count in channel_dict.items():
                    if channel not in aggregate_filter_stats[kernel]:
                        aggregate_filter_stats[kernel][channel] = 0
                    aggregate_filter_stats[kernel][channel] += count
        for kernel, channel_dict in aggregate_filter_stats.items():
            channel_dict = {k: v for k, v in sorted(channel_dict.items(), key=lambda item: item[1], reverse=True)}
            aggregate_filter_stats[kernel] = channel_dict
        return aggregate_filter_stats
    
    @ classmethod
    def get_aggregate_filter_stats(cls , stats_dict):
        aggregate_filter_stats = cls.get_aggregate_filter_stats_per_kernel(stats_dict)
        cross_kernel_aggregate_channel = {}
        for filter_dicts in aggregate_filter_stats.values():
            for filter_size, count in filter_dicts.items():
                if filter_size not in cross_kernel_aggregate_channel:
                    cross_kernel_aggregate_channel[filter_size] = 0
                cross_kernel_aggregate_channel[filter_size] += count
        cross_kernel_aggregate_filters = {k: v for k,v in sorted(cross_kernel_aggregate_channel.items(), key=lambda item: item[1], reverse=True)}
        return cross_kernel_aggregate_filters
        