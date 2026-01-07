import math
import warnings
from typing import Any, List, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .._buffer_dict import BufferDict
from .config import TuckAConfig


class TuckALayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("tucka_g", "tucka_c", "tucka_u")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = (
        "tucka_r",
        "tucka_k",
        "tucka_t",
        "tucka_p",
        "tucka_tensor_idx",
        "tucka_expert_weights",
    )

    def __init__(
        self,
        base_layer: nn.Module,
        tucka_tensor_idx: BufferDict,
        tucka_expert_weights: BufferDict,
        is_router: bool = False,
        tucka_config: TuckAConfig = None,
        **kwargs,
    ) -> None:
        # Orignal Layer & config
        self.base_layer = base_layer
        self.config = tucka_config

        # Routing
        self.is_router = is_router
        self.ec_initialized = False
        self.tucka_tensor_idx = tucka_tensor_idx
        self.tucka_expert_weights = tucka_expert_weights

        # Trainable
        self.tucka_g = nn.ParameterDict({})
        self.tucka_u = nn.ParameterDict({})
        self.tucka_c = nn.ParameterDict({})

        if self.is_router:
            self.tucka_ec = nn.ParameterDict({})
            self.adapter_layer_names = self.adapter_layer_names + ("tucka_ec",)

            # Configure ALB bias term
            if self.config.use_alb:
                self.tucka_alb_bias = nn.ParameterDict({})
                self.other_param_names = self.other_param_names + ("tucka_alb_bias",)

            # log expert load
            if self.config.log_expert_load != 0:
                self.expert_load = None
                self.logging_count = 0

        self._disable_adapters = False
        self.kwargs = kwargs

        # Set in & out feature sizes, and fan_in_fan_out depending on base layer type
        self.fan_in_fan_out = True
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = (
                base_layer.in_features,
                base_layer.out_features,
            )
        elif isinstance(base_layer, Conv1D):
            self.in_features, self.out_features = base_layer.nx, base_layer.nf
            self.fan_in_fan_out = False
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

    def update_layer(
        self,
        adapter_name: str,
        **kwargs,
    ) -> None:
        """Internal function to create tucka adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Overall rank for each expert.
            k (`int`): Number of independent tensor expert groups.
            t (`int`): Number of experts in each expert group.
            p (`int`): Shared rank between experts in a group.
        """
        r = self.config.r
        k = self.config.k
        t = self.config.t
        p = self.config.p

        if r <= 0 or k <= 0 or t <= 0 or p <= 0:
            raise ValueError(
                f"`r, k, t, p` should all be a positive integer value but the value passed is {r}, {k}, {t}, {p}"
            )

        # Determine shape of TuckA weights
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear) or isinstance(base_layer, Conv1D):
            self.tucka_g[adapter_name] = nn.Parameter(
                torch.empty(k, p, r, r), requires_grad=True
            )
            self.tucka_c[adapter_name] = nn.Parameter(
                torch.empty(t, p), requires_grad=True
            )
            self.tucka_u[adapter_name] = nn.Parameter(
                torch.empty(self.in_features, r), requires_grad=True
            )
            if self.is_router:
                self.tucka_ec[adapter_name] = nn.Parameter(
                    torch.empty(k * t, self.in_features), requires_grad=True
                )
                if self.config.use_alb:
                    self.tucka_expert_bias[adapter_name] = nn.Parameter(
                        torch.empty(k * t), requires_grad=False
                    )

                if self.config.log_expert_load != 0:
                    self.expert_load = [0] * k

        else:
            raise TypeError(
                f"TuckA is not implemented for base layers of type {type(base_layer).__name__}"
            )

        # Initialize weights
        self.reset_tucka_parameters_random(adapter_name)

        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_tucka_parameters_random(self, adapter_name: str):
        tucka_g_slice = torch.empty(self.config.r, self.config.r)
        nn.init.kaiming_uniform_(tucka_g_slice, a=math.sqrt(5))
        self.tucka_g[adapter_name].data.copy_(
            tucka_g_slice.unsqueeze(0)
            .unsqueeze(0)
            .repeat(self.config.k, self.config.p, 1, 1)
        )
        nn.init.kaiming_uniform_(
            self.tucka_u[adapter_name], mode="fan_out", a=math.sqrt(5)
        )
        nn.init.kaiming_uniform_(
            self.tucka_c[adapter_name], mode="fan_out", a=math.sqrt(5)
        )
        if self.is_router and (self.config.use_kaiming_init_ec or self.config.use_alb):
            nn.init.kaiming_uniform_(self.tucka_ec[adapter_name], a=math.sqrt(5))
            if self.config.use_alb:
                nn.init.zeros_(self.tucka_expert_bias[adapter_name])

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.tucka_g.keys():
                continue
            warnings.warn(
                "Scaling operation for TuckA not supported! Automatically set scale to 1."
            )

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.tucka_g.keys():
                continue
            warnings.warn(
                "Unscaling operation for TuckA not supported! Keeping scale at 1."
            )


class TuckALinear(nn.Module, TuckALayer):
    """
    TuckA implemented in a dense layer.
    """

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        tucka_config: TuckAConfig,
        tucka_tensor_idx: BufferDict,
        tucka_expert_weights: BufferDict,
        is_router: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        TuckALayer.__init__(
            self,
            base_layer,
            tucka_tensor_idx,
            tucka_expert_weights,
            is_router,
            tucka_config,
            **kwargs,
        )
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, **kwargs)

    def merge(self, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        In TuckA, merging the adapter with base weights is no longer possible due to dynamic routing

        Args:
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If `None`, all active adapters will be merged.
                Defaults to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        self.merged_trans = {}

    def _calc_adapter(self, x, tensor_idx, expert_weights, adapter_name):
        """
        This method passes the input through the tucka adapter.
        """
        g = self.tucka_g[adapter_name][tensor_idx]  # (p, r, r)
        u = self.tucka_u[adapter_name]  # (d, r)
        c = self.tucka_c[adapter_name]  # (t, p)

        g_norm = g / g.norm()
        u_norm = u / u.norm()
        c_norm = c / c.norm()

        # dtype & device check
        if g_norm.dtype != x.dtype:
            g_norm = g_norm.to(x.dtype)
            u_norm = u_norm.to(x.dtype)
            c_norm = c_norm.to(x.dtype)
        if expert_weights.device != x.device:
            expert_weights = expert_weights.to(x.device)

        x_u = torch.matmul(x, u_norm)  # x@u, shape: (..., r)
        cg = torch.einsum("tp,prs->trs", c_norm, g_norm)  # c@(mode1)g, shape: (t, r, r)
        try:
            mean_cg = torch.einsum(
                "t,trs->rs", expert_weights, cg
            )  # weighted mean, shape: (r, r)
        except Exception as e:
            raise RuntimeError(
                f"Possible router assignment error, try manually specifiy router_name in TuckAConfig.\n{e}"
            ) from e
        x_u_cg = torch.einsum(
            "...r,rs->...s", x_u, mean_cg
        )  # x@u@mean_cg, shape: (..., r)
        x_u_cg_u = torch.einsum(
            "...r,dr->...d", x_u_cg, u_norm
        )  # x@u@mean_cg@u^T, shape: (..., d)
        return x_u_cg_u

    def _initialize_centroids(self, x, active_adapter):
        in_features = x.size(1)
        perturbations = torch.rand(
            (self.config.k * self.config.t - 1, in_features),
            device=x.device,
            dtype=x.dtype,
        )
        perturbations = (perturbations - 0.5) * 2 * self.config.ec_perturb_scale

        # Construct the ec matrix
        centroids = [x]
        for i in range(self.config.k * self.config.t - 1):
            centroids.append(x + perturbations[i])
        centroids = torch.cat(centroids, dim=0)  # (num_experts, in_features)

        # Register as trainable
        self.tucka_ec[active_adapter].data.copy_(centroids)

    def _calc_affinity_score(self, x: torch.Tensor, active_adapter) -> torch.Tensor:
        input = x.mean(dim=0).mean(dim=0, keepdim=True)  # 1, in_features
        if not self.config.use_kaiming_init_ec and not self.ec_initialized:
            self._initialize_centroids(input.detach(), active_adapter)
        ec = self.tucka_ec[active_adapter]
        if ec.dtype != input.dtype:
            ec = ec.to(input.dtype)
        logits = F.linear(input, ec, None).squeeze(0)  # k*t
        logits = logits.view(self.config.k, self.config.t)  # (k, t)
        logits = logits / logits.norm(dim=1, keepdim=True)  # (k)
        logits = logits.view(-1)
        return logits.sigmoid()  # (t)

    def _calc_route(
        self, x: torch.Tensor, active_adapter
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This method is used only by the router layer to compute tensor_idx and expert_weights for this forward pass.
        """
        expert_score = self._calc_affinity_score(x, active_adapter)  # (t)
        _, top1_index = torch.topk(expert_score, 1, dim=-1)  # (topk)
        t = self.config.t
        tensor_idx = top1_index // t
        expert_weights = expert_score[
            tensor_idx * t : (tensor_idx + 1) * t
        ]  # Select the tensor containing the best slice

        # ----------------------logging------------------------
        if self.config.log_expert_load != 0:
            self.expert_load[tensor_idx] += 1

        return tensor_idx, expert_weights

    def _calc_route_alb(
        self, x: torch.Tensor, active_adapter
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Auxiliary Loss-free Balancing Router.
        """
        expert_score = self._calc_affinity_score(x, active_adapter)  # (t)
        _, top1_index = torch.topk(
            expert_score + self.tucka_expert_bias[active_adapter], 1, dim=-1
        )  # (topk)
        t = self.config.t
        tensor_idx = top1_index // t
        select_from = tensor_idx * t
        select_to = (tensor_idx + 1) * t

        # Update self.tucka_expert_bias[active_adapter] according to the index range
        bias = self.tucka_expert_bias[active_adapter]
        for idx in range(bias.shape[0]):
            if select_from <= idx < select_to:
                bias[idx] -= self.config.alb_gamma * (
                    self.config.k - 1
                )  # Ensure sum(bias) = 0
            else:
                bias[idx] += self.config.alb_gamma
        tensor_idx = top1_index // t
        expert_weights = expert_score[
            select_from:select_to
        ]  # Select the tensor containing the best slice
        expert_weights = expert_weights / expert_weights.sum()

        # ----------------------logging------------------------
        if self.config.log_expert_load != 0:
            self.expert_load[tensor_idx] += 1

        return tensor_idx, expert_weights

    def _avg_pairwise_mse(self, x):
        """
        Calculate the average pairwise MSE for a matrix.
        """
        with torch.no_grad():
            n, d = x.shape
            if n < 2:
                return torch.tensor(0.0, device=x.device, dtype=x.dtype)
            total_squared_norm = torch.sum(x**2)
            sum_vector = torch.sum(x, dim=0)
            squared_sum_vector = torch.sum(sum_vector**2)
            total_squared_dist = n * total_squared_norm - squared_sum_vector
            num_pairs = n * (n - 1) // 2
            avg_mse = total_squared_dist / (d * num_pairs)
            return avg_mse

    def _fetch_route(
        self, x: torch.Tensor, active_adapter
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Routing function.
        """
        if self.is_router:
            if self.config.use_alb and self.training:
                tensor_idx, expert_weights = self._calc_route_deepseek(
                    x, active_adapter
                )
            else:
                tensor_idx, expert_weights = self._calc_route(x, active_adapter)

            # -------------------------logging------------------------
            if self.config.log_expert_load != 0:
                if self.logging_count % self.config.log_expert_load == 0:
                    avg_mse = self._avg_pairwise_mse(
                        self.tucka_ec[active_adapter].data.detach().cpu()
                    )
                    print(f"\nExpert load: {self.expert_load}, Device: {x.device}")
                    print(f"Average pairwise MSE: {avg_mse.item()}, Device: {x.device}")
                    if self.config.use_alb:
                        print(
                            f"DeepSeek bias: {self.tucka_expert_bias[active_adapter].detach().cpu().numpy()}"
                        )
                    if self.config.dump_input_and_ec:
                        ec_np = self.tucka_ec[active_adapter].detach().cpu().numpy()
                        x_np = x.detach().cpu().mean(dim=1).numpy()
                        ec_path = f"./output/ec-{self.logging_count}.csv"
                        pd.DataFrame(ec_np).to_csv(ec_path, index=False, header=False)
                        x_path = f"./output/x-{self.logging_count}.csv"
                        pd.DataFrame(x_np).to_csv(x_path, index=False, header=False)
                self.logging_count += 1

            self.route = (tensor_idx, expert_weights)
            self.tucka_tensor_idx[active_adapter] = tensor_idx
            self.tucka_expert_weights[active_adapter] = expert_weights
        return (
            int(self.tucka_tensor_idx[active_adapter]),
            self.tucka_expert_weights[active_adapter],
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            return self.base_layer(x, *args, **kwargs)

        x = x.to(self.get_base_layer().weight.dtype)  # Cast dtype

        for active_adapter in self.active_adapters:
            if active_adapter not in self.tucka_g:
                continue
            tensor_idx, expert_weights = self._fetch_route(x, active_adapter)
            adapter_output = self._calc_adapter(
                x, tensor_idx, expert_weights, active_adapter
            )

        result = self.base_layer(
            x + self.config.alpha * adapter_output, *args, **kwargs
        )  # multiplication: x@(I+T)=x+x@T

        return result.to(previous_dtype)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "tucka." + rep
