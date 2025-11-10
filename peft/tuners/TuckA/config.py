# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from ast import dump
from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class TuckAConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`TuckAModel`].

    Args:
            r (`int`):
                The rank of TuckA expert across different layers.
            k (`int`):
                The number of tensor expert groups of TuckA across different layers.
            t (`int`):
                The number of experts in each group of TuckA across different layers.
            p (`int`):
                The rank shared among experts in each group of TuckA across different layers.
            alpha (`int`):
                The scaling factor for the adapter output. It is used to control the contribution of the adapter output to
                the final output.
            ec_perturb_scale (`float`):
                The perturb strength while initializing expert centroids. Only works with Data-Aware Init.
            log_expert_load (`int`):
                Print expert load every given number of forward steps. Defaults to 0, which means no output.
            use_kaiming_init_ec (`bool`):
                Whether to use Kaiming Uniform to initialize expert centroids. This is intended for running comparisons.
            use_alb (`bool`):
                Whether to use Auxiliary Loss-free Balancing for the router. This is intended for running comparisons.
            alb_gamma (`float`):
                The load balancing strength for ALB.
            dump_input_and_ec (`bool`):
                Whether to dump the input and expert centroids to CSV files for visualization.
            target_modules (`Optional[Union[list[str], str]]`):
                The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
                names will be replaced. When passing a string, a regex match will be performed. When passing a list of
                strings, either an exact match will be performed or it is checked if the name of the module ends with any
                of the passed strings. If this is specified as 'all-linear', then all linear modules are chosen, excluding
                the output layer. If this is not specified, modules will be chosen according to the model architecture. If
                the architecture is not known, an error will be raised -- in this case, you should specify the target
                modules manually.
            exclude_modules (`Optional[Union[list[str], str]]`):
                The names of the modules to not apply the adapter to. When passing a string, a regex match will be
                performed. When passing a list of strings, either an exact match will be performed or it is checked if the
                name of the module ends with any of the passed strings.
            init_weights (`bool`):
                Whether to perform initialization of TuckA weights.
            layers_to_transform (`Union[list[int], int]`):
                The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer indices
                that are specified in this list. If a single integer is passed, it will apply the transformations on the
                layer at this index.
            layers_pattern (`Optional[Union[list[str], str]]`):
                The layer pattern name, used only if `layers_to_transform` is different from `None`. This should target the
                `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`.
            bias (`str`):
                Bias type for TuckA. Can be one of 'none', 'all', or 'tucka_only'.
            modules_to_save (`list[str]`):
                List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
    """

    r: int = field(
        default=8,
        metadata={
            "help": "The rank of TuckA expert across different layers.",
        },
    )
    k: int = field(
        default=3,
        metadata={
            "help": "The number of tensor expert groups of TuckA across different layers.",
        },
    )
    t: int = field(
        default=3,
        metadata={
            "help": "The number of experts in each group of TuckA across different layers.",
        },
    )
    p: int = field(
        default=2,
        metadata={
            "help": "The rank shared among experts in each group of TuckA across different layers.",
        },
    )
    alpha: int = field(
        default=1,
        metadata={
            "help": "The scaling factor for the adapter output. It is used to control the contribution of the adapter output to the final output.",
        },
    )
    ec_perturb_scale: float = field(
        default=1,
        metadata={
            "help": "The perturb strength while initializing expert centroids. Only works with Data-Aware Init.",
        },
    )
    log_expert_load: int = field(
        default=0,
        metadata={
            "help": "Print expert load every given number of forward steps, defaluts to 0, which is no output",
        },
    )
    use_kaiming_init_ec: bool = field(
        default=False,
        metadata={
            "help": "Use Kaiming Uniform to initialize expert centroids, only for running comparisons.",
        },
    )
    use_alb: bool = field(
        default=False,
        metadata={
            "help": "Use Auxiliary Loss-free Balacing for router, only for running comparisons.",
        },
    )
    alb_gamma: float = field(
        default=1e-3,
        metadata={"help": "Load balancing strength for ALB"},
    )
    dump_input_and_ec: bool = field(
        default=False,
        metadata={
            "help": "Dump the input and expert centroids to csv files for visualization.",
        },
    )
    router_name: str = field(
        default="",
        metadata={
            "help": "Manually override the router assignment. You need to designate this if the model's module order is not identical to its forward flow.",
        },
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with TuckA.",
            "example": "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' ",
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to exclude from TuckA."
        },
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the TuckA layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern. "
            "This should target the `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`."
        },
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for TuckA. Can be 'none', 'all' or 'tucka_only'"},
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from TuckA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = "TUCKA"
        self.target_modules = (
            set(self.target_modules)
            if isinstance(self.target_modules, list)
            else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules)
            if isinstance(self.exclude_modules, list)
            else self.exclude_modules
        )
        # if target_modules is a regex expression, then layers_to_transform should be None
        if (
            isinstance(self.target_modules, str)
            and self.layers_to_transform is not None
        ):
            raise ValueError(
                "`layers_to_transform` cannot be used when `target_modules` is a str."
            )

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError(
                "`layers_pattern` cannot be used when `target_modules` is a str."
            )

        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError(
                "When `layers_pattern` is specified, `layers_to_transform` must also be specified. "
            )
