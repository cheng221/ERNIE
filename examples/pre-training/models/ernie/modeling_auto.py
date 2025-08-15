# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Paddle Ernie model"""
import math
from functools import partial
import logging
from typing import Optional, Tuple
import contextlib
import inspect

from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker
from paddle.distributed import in_auto_parallel_align_mode

from models.comm_utils import subbatch

from models.moe.top2_gate_auto import Top2Gate
from models.moe.top2_gate_auto import TopKGateFusedAuto


from paddleformers.transformers.conversion_utils import (
    StateDictNameMapping,
    init_name_mappings,
)

from paddleformers.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions as _BaseModelOutput,
)
from paddleformers.transformers.model_outputs import CausalLMOutputWithCrossAttentions

from paddleformers.transformers.model_utils import PretrainedModel, register_base_model

from models.ernie.modeling import (
    FusedDropoutImpl,
    RotaryEmbedding,
    RMSNorm,
    get_triangle_upper_mask,
    mem_eff_attn,
    inbatch_pack_offset_to_attn_mask_start_row_indices,
    _make_causal_mask,
    _expand_mask,
)
from models.ernie.modeling_moe import (
    ErnieMoeMLPFused,
)
from models.sequence_parallel_utils_auto import (
    sequence_parallel_sparse_mask_labels,
)
from models.moe.moe_layer_auto import (
    MOELayerAuto,
)
from .configuration import ErnieMoEConfig
from models.moe.moe_utils_auto import get_mesh


@dataclass
class BaseModelOutputWithPastAndCrossAttentions(_BaseModelOutput):

    router_loss: Optional[paddle.Tensor] = None
    gate_logits: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class CausalLMOutputWithCrossAttentionsAuto(CausalLMOutputWithCrossAttentions):

    router_loss: Optional[paddle.Tensor] = None


logger = logging.getLogger(__name__)


try:
    from paddle.nn.functional.flash_attention import flash_attention

    logger.warning(
        "Use flash attention in scaled-dot-product. Attention mask is deprecated"
    )
except (ImportError, ModuleNotFoundError):
    flash_attention = None

try:
    from paddle.nn.functional.flash_attention import flash_attention_with_mask
except (ImportError, ModuleNotFoundError):
    try:
        from paddle.nn.functional.flash_attention import (
            scaled_dot_product_attention as flash_attention_with_mask,
        )
    except (ImportError, ModuleNotFoundError):
        logger.warning(
            "flash_attention_with_mask not found. Use FleetY8.2 SFT instead."
        )
        flash_attention_with_mask = None

try:
    from paddle.nn.functional.flash_attention import flash_attention_with_sparse_mask
except (ImportError, ModuleNotFoundError):
    logger.warning("flash_attention_with_sparse_mask not found. Use FleetY8.9 instead.")
    flash_attention_with_sparse_mask = None

try:
    from to_block_diag_causal_mask import to_block_diag_causal_mask
except (ImportError, ModuleNotFoundError):
    logger.warning("to_block_diag_causal_mask not found. Use FleetY8.2 SFT instead.")
    to_block_diag_causal_mask = None

try:
    from fast_ln import fast_ln
except ImportError:
    logger.warning(
        "fast-ln not found, run `python src/ops/fast_ln_setup.py install` to build fast ln"
    )
    fast_ln = None

try:
    from paddle.incubate.nn.functional import (
        fused_rotary_position_embedding as fused_rope,
    )
except (ImportError, ModuleNotFoundError):
    logger.warning("fused_rotary_position_embedding not found")
    fused_rope = None

try:
    from paddle.incubate.nn.functional import swiglu as fused_swiglu
except (ImportError, ModuleNotFoundError):
    fused_swiglu = None


ERNIE_PRETRAINED_MODEL_ARCHIVE_LIST = []

__all__ = [
    "ErnieModelAuto",
    "ErniePretrainedModelAuto",
    "ErnieForCausalLMAuto",
]


gate_class = dict(
    top2=Top2Gate,
    top2_fused=TopKGateFusedAuto,
)


def global_mesh_starts_with_pp():

    mesh = fleet.auto.get_mesh()
    if "pp" in mesh.dim_names:
        return mesh.get_mesh_with_dim("pp")
    else:
        return mesh


def is_fleety_func():
    """
    Check whether it is PaddlePaddle FleetY version.
    """
    if flash_attention_with_sparse_mask is None:
        return True

    args = inspect.getfullargspec(flash_attention_with_sparse_mask).args
    return "causal" in args


IS_FLEETY = is_fleety_func()


def calc_lm_head_logits(
    config,
    hidden_states,
    weight,
    bias,
    sparse_label_idx=None,
    tensor_parallel_output=None,
):
    """the core function to calc lm head"""
    if config.sequence_parallel:

        assert (
            not config.use_sparse_head_and_loss_fn
        ), "use_sparse_head_and_loss_fn is not supported now."

        hcg = paddle.distributed.fleet.get_hybrid_communicate_group()
        dp_rank = hcg.get_data_parallel_rank()
        sharding_rank = hcg.get_sharding_parallel_rank()
        if dp_rank <= 1 and sharding_rank <= 1:
            hidden_states = dist.reshard(
                hidden_states,
                get_mesh(-1),
                [dist.Replicate(), dist.Replicate()],
            )
        else:
            hidden_states = dist.reshard(
                hidden_states,
                get_mesh(-1),
                [dist.Shard(1), dist.Replicate()],
            )
        hidden_states = paddle.transpose(hidden_states, [1, 0, 2])
        if not config.using_dynamic_sequence_length:
            hidden_states = hidden_states.reshape(
                [-1, config.seqlen, hidden_states.shape[-1]]
            )
        else:
            assert (
                config.micro_batch_size
            ), "micro_batch_size should be set when using dygramic sequence length."
            hidden_states = hidden_states.reshape(
                [config.micro_batch_size, -1, hidden_states.shape[-1]]
            )
    if tensor_parallel_output is None:
        tensor_parallel_output = config.tensor_parallel_output
    logits = paddle.matmul(
        hidden_states, weight, transpose_y=config.tie_word_embeddings
    )
    if bias is not None:
        logits += bias

    if config.tensor_parallel_degree > 1 and not tensor_parallel_output:
        logits = dist.reshard(logits, get_mesh(-1), [dist.Shard(0), dist.Replicate()])

    return logits


def scaled_dot_product_attention(
    query_states,
    key_states,
    value_states,
    attention_mask,
    output_attentions,
    config,
    is_causal=True,
    rr_flash_attn=None,
    inbatch_pack_offset=None,
    training=True,
):

    bsz, q_len, num_heads, head_dim = query_states.shape
    _, kv_seq_len, num_key_value_heads, _ = value_states.shape

    can_use_fa = config.use_flash_attn and flash_attention is not None
    can_use_fa_sparse_mask = (
        config.use_mem_eff_attn
        and inbatch_pack_offset is not None
        and flash_attention_with_sparse_mask is not None
    )

    if not can_use_fa and not can_use_fa_sparse_mask:
        if query_states.shape[-2] != key_states.shape[-2]:
            key_states = key_states.repeat_interleave(
                num_heads // num_key_value_heads, axis=-2
            )
        if query_states.shape[-2] != value_states.shape[-2]:
            value_states = value_states.repeat_interleave(
                num_heads // num_key_value_heads, axis=-2
            )

    if can_use_fa:
        if rr_flash_attn is not None:
            attn_output, attn_weights = rr_flash_attn(
                query_states,
                key_states,
                value_states,
                dropout=config.attention_probs_dropout_prob,
                causal=is_causal and query_states.shape[1] != 1,
                return_softmax=output_attentions,
            )
        else:
            attn_output, attn_weights = flash_attention(
                query_states,
                key_states,
                value_states,
                dropout=config.attention_probs_dropout_prob,
                causal=is_causal and query_states.shape[1] != 1,
                return_softmax=output_attentions,
            )

        attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
        return attn_output, attn_weights
    elif config.use_mem_eff_attn and inbatch_pack_offset is not None:
        assert (
            not output_attentions
        ), "output_attentions should be False when use_mem_eff_attn=True"
        if config.use_flash_attn_with_mask:
            if flash_attention_with_sparse_mask is not None:
                causal_mask_indices, attn_mask_min_start_row = (
                    inbatch_pack_offset_to_attn_mask_start_row_indices(
                        inbatch_pack_offset
                    )
                )
                if IS_FLEETY:
                    kwargs = {
                        "causal": True,
                        "dropout": config.attention_probs_dropout_prob,
                    }
                else:
                    kwargs = {
                        "is_causal": True,
                        "dropout_p": config.attention_probs_dropout_prob,
                    }
                attn_output = flash_attention_with_sparse_mask(
                    query_states.astype(value_states.dtype),
                    key_states.astype(value_states.dtype),
                    value_states.astype(value_states.dtype),
                    attn_mask_start_row_indices=causal_mask_indices,
                    attn_mask_start_row=attn_mask_min_start_row,
                    **kwargs,
                )
            else:
                attn_mask = to_block_diag_causal_mask(
                    inbatch_pack_offset, q_len, float("-inf"), "bfloat16"
                )
                attn_output = flash_attention_with_mask(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask,
                    config.attention_probs_dropout_prob,
                )
        else:
            attn_output = mem_eff_attn(
                query_states,
                key_states,
                value_states,
                inbatch_pack_offset,
                drop_prob=config.attention_probs_dropout_prob,
            )
        attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
        return attn_output, None
    else:

        query_states = paddle.transpose(query_states, [0, 2, 1, 3]) / math.sqrt(
            head_dim
        )
        key_states = paddle.transpose(key_states, [0, 2, 1, 3])
        value_states = paddle.transpose(value_states, [0, 2, 1, 3])

        attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2]))

        if attn_weights.shape != [bsz, num_heads, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention weights should be of shape {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is None:
            attention_mask = get_triangle_upper_mask(attn_weights)

        attention_mask = attention_mask.reshape([bsz, 1, q_len, kv_seq_len])
        if attention_mask.shape != [bsz, 1, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention mask should be of shape {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
            )
        if training:
            attn_weights = attention_mask + attn_weights
            attn_weights = paddle.maximum(
                attn_weights,
                paddle.to_tensor(
                    float(paddle.finfo(query_states.dtype).min),
                    dtype=query_states.dtype,
                ),
            )

            if paddle.in_dynamic_mode():
                with paddle.amp.auto_cast(False):
                    attn_weights = F.softmax(
                        attn_weights, axis=-1, dtype="float32"
                    ).astype(query_states.dtype)
            else:
                attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(
                    query_states.dtype
                )
        else:
            attn_weights = attn_weights.cast(paddle.float32)
            attention_mask = attention_mask.cast(paddle.float32)
            attn_weights = attn_weights.add_(attention_mask)
            attn_weights = F.softmax_(attn_weights, axis=-1).astype(query_states.dtype)

        if config.attention_probs_dropout_prob > 0.0:
            if config.tensor_parallel_degree > 1:
                with get_rng_state_tracker().rng_state("local_seed"):
                    attn_weights = F.dropout(
                        attn_weights,
                        config.attention_probs_dropout_prob,
                        training=training,
                        mode="upscale_in_train",
                    )
            else:
                attn_weights = F.dropout(
                    attn_weights,
                    config.attention_probs_dropout_prob,
                    training=training,
                    mode="upscale_in_train",
                )

        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
        if output_attentions:
            return attn_output, attn_weights
        return attn_output, None


def get_gate(
    config: ErnieMoEConfig,
    expert: Tuple[Tuple[int, nn.Layer]],
    layer_idx: int,
    ipp: int = 0,
) -> Tuple[nn.Layer, nn.LayerList]:

    moe_num_experts = config.moe_num_experts
    assert (
        moe_num_experts >= config.moe_world_size
    ), f"expert moe_num_experts={moe_num_experts} >= moe_world_size={config.moe_world_size}"
    assert (
        moe_num_experts % config.moe_world_size == 0
    ), f"expert moe_num_experts={moe_num_experts} % moe_world_size={config.moe_world_size} == 0"
    moe_num_experts_per_device = moe_num_experts // config.moe_world_size
    experts = nn.LayerList([])
    for expert_id, (experts_num, fc) in enumerate(expert):
        assert experts_num % config.moe_world_size == 0
        experts_to_append = []
        if not hasattr(fc, "__len__"):
            experts_to_append.append(fc)
            if expert_id == 1:
                with paddle.utils.unique_name.guard("_mm_deepcopy"):
                    for _ in range(experts_num - 1):
                        experts_to_append.append(deepcopy(fc))
            else:
                for _ in range(experts_num - 1):
                    experts_to_append.append(deepcopy(fc))
        else:
            experts_to_append = fc
        for ex in experts_to_append:
            for p in ex.parameters():
                p.expert_type = f"expert_type_{expert_id}"
        experts.extend(experts_to_append)

    logger.info(
        f"using moe-world-size: {config.moe_world_size} "
        f"expert-per-device: {moe_num_experts_per_device} "
    )
    if config.moe_use_hard_gate and moe_num_experts <= 2:
        gate = None
        logger.info("MOE-GATE:-hard-gate")
    else:
        logger.info(f"MOE-GATE:-{config.moe_gate}")
        gate = gate_class[config.moe_gate.lower()](
            config, layer_idx=layer_idx, group=config.moe_group, ipp=ipp
        )

    index = 0 if config.moe_group == "dp" else 1
    ep_sub_meshes = dist.auto_parallel.api.split_mesh(get_mesh(ipp), index)

    for i, expert in enumerate(experts):
        ep_group_id = i // moe_num_experts_per_device
        if isinstance(expert, (ErnieMoeMLPFused, ErnieMoeMLP)):
            experts[i].redistribute_expert(
                ep_sub_meshes[ep_group_id], [dist.Replicate(), dist.Replicate()]
            )
            experts[i].ep_group_id = ep_group_id

    return gate, experts


class FastLayerNorm(nn.LayerNorm):
    def __init__(self, config):
        assert fast_ln is not None
        super().__init__(config.hidden_size, epsilon=config.rms_norm_eps)

    def forward(self, hidden_states):
        return fast_ln(hidden_states, self.weight, self.bias, self._epsilon)[0]


class ErnieMLP(nn.Layer):

    def __init__(self, config, ipp=None, do_shard_tensor=True):
        super().__init__()
        self.config = config
        self.ipp = ipp
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias_attr=config.use_bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias_attr=config.use_bias
        )

        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias_attr=config.use_bias
        )

        if do_shard_tensor and (
            self.config.tensor_parallel_degree > 1
            or self.config.pipeline_parallel_degree > 1
        ):
            self.gate_proj.weight = dist.shard_tensor(
                self.gate_proj.weight,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Shard(1)],
            )
            self.up_proj.weight = dist.shard_tensor(
                self.up_proj.weight,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Shard(1)],
            )
            if config.use_bias:
                self.gate_proj.bias = dist.shard_tensor(
                    self.gate_proj.bias,
                    get_mesh(self.ipp),
                    [dist.Replicate(), dist.Shard(0)],
                )
                self.up_proj.bias = dist.shard_tensor(
                    self.up_proj.bias,
                    get_mesh(self.ipp),
                    [dist.Replicate(), dist.Shard(0)],
                )
            self.down_proj.weight = dist.shard_tensor(
                self.down_proj.weight,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Shard(0)],
            )
            if config.use_bias:
                self.down_proj.bias = dist.shard_tensor(
                    self.down_proj.bias,
                    get_mesh(self.ipp),
                    [dist.Replicate(), dist.Replicate()],
                )

        self.fuse_swiglu = config.fuse_swiglu
        if self.fuse_swiglu:
            assert fused_swiglu is not None, "fused_swiglu operator is not found."

    def forward(self, x):

        if self.fuse_swiglu:
            x = fused_swiglu(self.gate_proj(x), self.up_proj(x))
        else:
            x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        out = self.down_proj(x)
        if self.config.sequence_parallel:
            out = dist.reshard(out, get_mesh(self.ipp), [dist.Shard(1), dist.Shard(0)])
        return out


class ErnieAttentionAuto(nn.Layer):

    def __init__(self, config, ipp: Optional[int] = None):
        super().__init__()
        self.ipp = ipp
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.use_recompute_attn = config.use_recompute_attn
        self.is_gqa = (
            config.num_key_value_heads is not None
            and config.num_key_value_heads != self.num_heads
        )
        if config.fuse_rope:
            assert fused_rope is not None, "fused_rope is not supported"
        self.fuse_rope = config.fuse_rope

        if self.is_gqa:
            logger.info(
                f"use GQA - num_heads: {self.num_heads}- num_key_value_heads: {self.num_key_value_heads}"
            )
            assert (
                self.num_heads % self.num_key_value_heads == 0
            ), f"num_heads: {self.num_heads}, num_key_value_heads: {self.num_key_value_heads}"
            kv_hidden_size = (
                self.hidden_size // self.num_heads * self.num_key_value_heads
            )

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias_attr=config.use_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size if not self.is_gqa else kv_hidden_size,
            bias_attr=config.use_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size if not self.is_gqa else kv_hidden_size,
            bias_attr=config.use_bias,
        )

        self.o_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias_attr=config.use_bias,
        )

        self.config = config

        if (
            self.config.tensor_parallel_degree > 1
            or self.config.pipeline_parallel_degree > 1
        ):
            self.q_proj.weight = dist.shard_tensor(
                self.q_proj.weight,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Shard(1)],
            )
            self.k_proj.weight = dist.shard_tensor(
                self.k_proj.weight,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Shard(1)],
            )
            self.v_proj.weight = dist.shard_tensor(
                self.v_proj.weight,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Shard(1)],
            )
            if config.use_bias:
                self.q_proj.bias = dist.shard_tensor(
                    self.q_proj.bias,
                    get_mesh(self.ipp),
                    [dist.Replicate(), dist.Shard(0)],
                )
                self.k_proj.bias = dist.shard_tensor(
                    self.k_proj.bias,
                    get_mesh(self.ipp),
                    [dist.Replicate(), dist.Shard(0)],
                )
                self.v_proj.bias = dist.shard_tensor(
                    self.v_proj.bias,
                    get_mesh(self.ipp),
                    [dist.Replicate(), dist.Shard(0)],
                )
            self.o_proj.weight = dist.shard_tensor(
                self.o_proj.weight,
                get_mesh(self.ipp),
                [dist.Replicate(), dist.Shard(0)],
            )

    def forward(
        self,
        hidden_states,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        inbatch_pack_offset: Optional[Tuple[paddle.Tensor]] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        if self.config.sequence_parallel:
            hidden_states = dist.reshard(
                hidden_states, get_mesh(self.ipp), [dist.Shard(1), dist.Replicate()]
            )

        query_states = self.q_proj(hidden_states).reshape(
            shape=[0, 0, self.num_heads, self.head_dim]
        )
        key_states = self.k_proj(hidden_states).reshape(
            shape=[
                0,
                0,
                self.num_key_value_heads if self.is_gqa else self.num_heads,
                self.head_dim,
            ]
        )
        value_states = self.v_proj(hidden_states).reshape(
            shape=[
                0,
                0,
                self.num_key_value_heads if self.is_gqa else self.num_heads,
                self.head_dim,
            ]
        )

        if self.config.sequence_parallel:
            query_states = paddle.transpose(query_states, [1, 0, 2, 3])
            key_states = paddle.transpose(key_states, [1, 0, 2, 3])
            value_states = paddle.transpose(value_states, [1, 0, 2, 3])

        if self.use_recompute_attn:
            assert past_key_value is None, "do not use kv cache in recompute"
            assert not use_cache
            attn_output, attn_weights, past_key_value = recompute(
                self.rope_attn,
                None,
                query_states,
                key_states,
                value_states,
                attention_mask,
                position_ids,
                output_attentions,
                past_key_value,
                use_cache,
                inbatch_pack_offset,
                use_reentrant=False,
            )
        else:
            attn_output, attn_weights, past_key_value = self.rope_attn(
                mix_layer=None,
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                past_key_value=past_key_value,
                use_cache=use_cache,
                inbatch_pack_offset=inbatch_pack_offset,
            )

        if self.config.sequence_parallel:
            attn_output = paddle.transpose(attn_output, [1, 0, 2])

        attn_output = self.o_proj(attn_output)
        if self.config.sequence_parallel:
            attn_output = dist.reshard(
                attn_output, get_mesh(self.ipp), [dist.Shard(1), dist.Shard(0)]
            )

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def rope_attn(
        self,
        mix_layer,
        query_states,
        key_states,
        value_states,
        attention_mask,
        position_ids,
        output_attentions=False,
        past_key_value=None,
        use_cache=False,
        inbatch_pack_offset=None,
    ):
        if mix_layer is not None:
            query_states, key_states, value_states = paddle.split(mix_layer, 3, axis=-1)
        query_states_dtype = query_states.dtype

        kv_seq_len = key_states.shape[-3]
        offset = 0
        if past_key_value is not None:
            offset = past_key_value[0].shape[-3]
            kv_seq_len += offset

        if self.config.rope_reorder:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

            query_states, key_states = self.rotary_emb.apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                position_ids=position_ids,
                offset=offset if position_ids is None else 0,
            )
        else:
            if offset > 0 or position_ids is not None or not self.fuse_rope:
                cos_sin = self.rotary_emb(kv_seq_len, position_ids).transpose(
                    [0, 2, 1, 3]
                )
                if offset > 0 and position_ids is None:
                    cos_sin = cos_sin[:, offset:]
                query_states, key_states = self.rotary_emb.apply_rotary(
                    cos_sin, query_states, key_states
                )
            else:
                bsz, q_len, num_heads, head_dim = query_states.shape
                _, kv_seq_len, num_key_value_heads, _ = key_states.shape
                if num_heads != num_key_value_heads:
                    query_states, _, _ = fused_rope(query_states, None, None)
                    key_states, _, _ = fused_rope(key_states, None, None)
                else:
                    query_states, key_states, _ = fused_rope(
                        query_states, key_states, None
                    )

        if use_cache:
            query_states = query_states.astype(query_states_dtype)
            key_states = key_states.astype(query_states_dtype)
        if past_key_value is not None:
            key_states = paddle.concat([past_key_value[0], key_states], axis=1)
            value_states = paddle.concat([past_key_value[1], value_states], axis=1)

        past_key_value = [key_states, value_states] if use_cache else None

        attn_output, attn_weights = scaled_dot_product_attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            config=self.config,
            inbatch_pack_offset=inbatch_pack_offset,
            training=self.training,
        )
        return attn_output, attn_weights, past_key_value


class ErnieMoeMLP(ErnieMLP):
    """_summary_

    Args:
        ErnieMoeMLP (_type_): _description_
    """

    def __init__(self, config, ipp=0):
        """
        doc
        """
        disable_ffn_model_parallel = getattr(
            config, "disable_ffn_model_parallel", False
        )
        if disable_ffn_model_parallel:
            config = deepcopy(config)
            config.tensor_parallel_degree = 1
            config.sequence_parallel = False

        super().__init__(config, ipp, do_shard_tensor=not disable_ffn_model_parallel)
        self.moe_dropout_prob = config.moe_dropout_prob
        self.fuse_swiglu = config.fuse_swiglu
        if self.fuse_swiglu:
            assert fused_swiglu is not None, "fused_swiglu operator is not found."

    def redistribute_expert(self, mesh, placements):
        """
        Place the experts on different devices.
        """
        self.gate_proj.weight = dist.shard_tensor(
            self.gate_proj.weight, mesh, placements
        )
        self.up_proj.weight = dist.shard_tensor(self.up_proj.weight, mesh, placements)
        self.down_proj.weight = dist.shard_tensor(
            self.down_proj.weight, mesh, placements
        )
        if self.config.use_bias:
            self.gate_proj.bias = dist.shard_tensor(
                self.gate_proj.bias, mesh, placements
            )
            self.up_proj.bias = dist.shard_tensor(self.up_proj.bias, mesh, placements)
            self.down_proj.bias = dist.shard_tensor(
                self.down_proj.bias, mesh, placements
            )

    def forward(self, x):

        if self.fuse_swiglu:
            x = fused_swiglu(self.gate_proj(x), self.up_proj(x))
        else:
            x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        if self.moe_dropout_prob > 0:
            with get_rng_state_tracker().rng_state("local_seed"):
                x = F.dropout(x=x, p=self.moe_dropout_prob)
        ret = self.down_proj(x)
        return ret


class BMMLinear(nn.Layer):

    def __init__(self, experts, d_in, d_out, use_bias=False):
        super().__init__()
        self.weight = self.create_parameter(
            [experts, d_in, d_out], dtype=paddle.get_default_dtype()
        )
        if use_bias:
            self.bias = self.create_parameter(
                [experts, d_out], dtype=paddle.get_default_dtype(), is_bias=True
            )
        else:
            self.bias = None

    def forward(self, x):
        """x: [num_experts, Seq, dim]"""
        if self.bias is not None:
            return paddle.bmm(x, self.weight) + self.bias
        return paddle.bmm(x, self.weight)


class ErnieDecoderLayerAuto(nn.Layer):
    """
    ErnieDecoderLayerAuto is a decoder layer in Ernie model.
    It is composed of self-attention, cross-attention and feedforward layers.
    """

    def __init__(self, config, layer_idx=0, ipp=0):
        """
            Initializes the ErnieBlock module.

        Args:
            config (ErnieConfig): The model configuration.
            layer_idx (int, optional): The index of this block in the model. Defaults to 0.
            ipp (int, optional): The index of this block in the pipeline parallelism. Defaults to 0.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.ipp = ipp
        self.hidden_size = config.hidden_size
        self.self_attn = ErnieAttentionAuto(config, ipp)
        self.use_moe = config.use_moe if hasattr(config, "use_moe") else False
        if self.use_moe:
            moe_layer_start_index = (
                min(config.moe_layer_start_index)
                if isinstance(config.moe_layer_start_index, (tuple, list))
                else config.moe_layer_start_index
            )
            moe_layer_end_index = (
                max(config.moe_layer_end_index)
                if isinstance(config.moe_layer_end_index, (tuple, list))
                else config.moe_layer_end_index
            )

        if (
            self.use_moe
            and ((layer_idx + 1) % config.moe_layer_interval == 0)
            and layer_idx >= moe_layer_start_index
            and layer_idx <= moe_layer_end_index
        ):
            self.create_moe_mlp_layer(layer_idx, ipp)
        else:
            self.mlp = ErnieMLP(config, ipp)
        if config.use_rmsnorm:
            Norm = RMSNorm(config)
        elif config.use_fast_ln:
            Norm = FastLayerNorm(config)
        else:
            Norm = nn.LayerNorm(config.hidden_size, epsilon=config.rms_norm_eps)
        if config.pipeline_parallel_degree > 1:
            Norm.weight = dist.shard_tensor(
                Norm.weight, get_mesh(ipp), [dist.Replicate(), dist.Replicate()]
            )
            if hasattr(Norm, "bias"):
                Norm.bias = dist.shard_tensor(
                    Norm.bias, get_mesh(ipp), [dist.Replicate(), dist.Replicate()]
                )

        self.input_layernorm = Norm
        self.post_attention_layernorm = Norm
        self.residual_add1 = FusedDropoutImpl(
            config.hidden_dropout_prob, mode="upscale_in_train"
        )
        self.residual_add2 = FusedDropoutImpl(
            config.hidden_dropout_prob, mode="upscale_in_train"
        )

    def create_moe_mlp_layer(self, layer_idx, ipp):
        _ex_cfg = deepcopy(self.config)
        fc_cls = ErnieMoeMLPFused if _ex_cfg.moe_fuse_experts else ErnieMoeMLP
        if _ex_cfg.moe_intermediate_size:
            if isinstance(_ex_cfg.moe_intermediate_size, (tuple, list)):
                assert isinstance(_ex_cfg.moe_num_experts, (tuple, list)) and len(
                    _ex_cfg.moe_num_experts
                ) == len(_ex_cfg.moe_intermediate_size)
                fc = []
                for _i, (num_experts, intermediate_size) in enumerate(
                    zip(_ex_cfg.moe_num_experts, _ex_cfg.moe_intermediate_size)
                ):
                    _ex_cfg_real = deepcopy(_ex_cfg)
                    _ex_cfg_real.intermediate_size = intermediate_size
                    cur_modality_start_layer_idx = (
                        self.config.moe_layer_start_index[_i]
                        if isinstance(self.config.moe_layer_start_index, (tuple, list))
                        else self.config.moe_layer_start_index
                    )
                    cur_modality_end_layer_idx = (
                        self.config.moe_layer_end_index[_i]
                        if isinstance(self.config.moe_layer_end_index, (tuple, list))
                        else self.config.moe_layer_end_index
                    )
                    if (
                        layer_idx >= cur_modality_start_layer_idx
                        and layer_idx <= cur_modality_end_layer_idx
                    ):
                        if _i == 1:
                            with paddle.utils.unique_name.guard(
                                f"mm_expert_{layer_idx}_"
                            ):
                                fc.append((num_experts, fc_cls(_ex_cfg_real)))
                        else:
                            fc.append((num_experts, fc_cls(_ex_cfg_real)))
                    else:
                        logger.info(
                            f"moe multimodal experts use Identity layer_idx: {layer_idx}"
                        )
                        fc.append((num_experts, nn.Identity()))
            else:
                _ex_cfg.intermediate_size = _ex_cfg.moe_intermediate_size
                fc = [(_ex_cfg.moe_num_experts, fc_cls(_ex_cfg))]
        else:
            fc = [(_ex_cfg.moe_num_experts, fc_cls(_ex_cfg))]
        gate, experts = get_gate(self.config, fc, layer_idx, self.ipp)
        _sh_cfg = deepcopy(self.config)

        if _sh_cfg.moe_num_shared_experts > 0:
            if _sh_cfg.moe_intermediate_size:
                _sh_inter_size = (
                    _sh_cfg.moe_intermediate_size[0]
                    if isinstance(_sh_cfg.moe_intermediate_size, (tuple, list))
                    else _sh_cfg.moe_intermediate_size
                )
                _sh_cfg.intermediate_size = (
                    _sh_inter_size * _sh_cfg.moe_num_shared_experts
                )
            else:
                _sh_cfg.intermediate_size = (
                    _sh_cfg.intermediate_size * _sh_cfg.moe_num_shared_experts
                )
            _sh_cfg.disable_ffn_model_parallel = False
            shared_experts = ErnieMoeMLP(_sh_cfg, ipp)
        else:
            shared_experts = None

        is_moe_infer = self.config.get("is_moe_infer", False)
        if is_moe_infer:
            raise NotImplementedError
        elif self.config.moe_use_size_all2all:
            raise NotImplementedError
        else:
            logger.info(f"moe-logging:{self.config.moe_logging}")
            moe_cls = MOELayerAuto
            self.mlp = moe_cls(
                gate,
                experts,
                layer_idx=layer_idx,
                shared_experts=shared_experts,
                group=self.config.moe_group,
                recompute=self.config.use_recompute_moe,
                enable_logging=self.config.moe_logging,
                k=self.config.moe_k,
                enable_pbr=self.config.moe_use_bpr,
                all_to_all_dropout=self.config.moe_all_to_all_dropout,
                group_experts=self.config.moe_group_experts,
                config=self.config,
                ipp=self.ipp,
            )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        inbatch_pack_offset: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        output_gate_logits=True,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """
        Args:
            hidden_states (`paddle.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`paddle.Tensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `cache` key value states are returned and can be used to speed up decoding
                (see `cache`).
            cache (`Tuple(paddle.Tensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        (hidden_states, self_attn_weights, present_key_value, *router_loss_attn) = (
            self.self_attn(
                hidden_states=hidden_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                use_cache=use_cache,
                inbatch_pack_offset=inbatch_pack_offset,
            )
        )

        if (
            self.config.tensor_parallel_degree > 1
            and self.config.hidden_dropout_prob > 0.0
        ):
            current_seed = (
                "local_seed" if self.config.sequence_parallel else "global_seed"
            )
            with get_rng_state_tracker().rng_state(current_seed):
                hidden_states = self.residual_add1(hidden_states, residual)
        else:
            hidden_states = self.residual_add1(hidden_states, residual)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if isinstance(
            self.mlp,
            (MOELayerAuto),
        ):

            hidden_states, _, router_loss, gate_logits = self.mlp(
                hidden_states, token_type_ids
            )
        else:
            if self.config.sequence_parallel:
                hidden_states = dist.reshard(
                    hidden_states,
                    get_mesh(self.ipp),
                    [dist.Shard(1), dist.Replicate()],
                )
            hidden_states = self.mlp(hidden_states)
            gate_logits = None

        if (
            self.config.tensor_parallel_degree > 1
            and self.config.hidden_dropout_prob > 0.0
        ):
            current_seed = (
                "local_seed" if self.config.sequence_parallel else "global_seed"
            )
            with get_rng_state_tracker().rng_state(current_seed):
                hidden_states = self.residual_add2(hidden_states, residual)
        else:
            hidden_states = self.residual_add2(hidden_states, residual)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if hasattr(self.config, "use_moe") and self.config.use_moe:
            if router_loss_attn:
                router_loss_attn = router_loss_attn[0]
                router_loss = router_loss + router_loss_attn

            if isinstance(self.mlp, (MOELayerAuto)):
                outputs += (router_loss,)
            else:
                outputs += (paddle.zeros([1], dtype=paddle.float32),)

            if output_gate_logits:
                outputs += (gate_logits,)

        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]
        return outputs


class ErniePretrainedModelAuto(PretrainedModel):
    """
    ErniePretrainedModelAuto is a pretrained model class for Ernie model.
    It is composed of a encoder and a decoder.
    """

    config_class = ErnieMoEConfig
    base_model_prefix = "ernie"

    @classmethod
    def _get_name_mappings(cls, config: ErnieMoEConfig) -> StateDictNameMapping:

        mappings: StateDictNameMapping = []
        model_mappings = [
            ["embed_tokens.weight"],
            ["norm.weight"],
        ]
        for layer_index in range(
            config.num_hidden_layers
            if not config.remove_tail_layer
            else config.num_hidden_layers - 1
        ):
            layer_mappings = [
                [
                    f"layers.{layer_index}.self_attn.q_proj.weight",
                    None,
                    "transpose",
                ],
                [
                    f"layers.{layer_index}.self_attn.k_proj.weight",
                    None,
                    "transpose",
                ],
                [
                    f"layers.{layer_index}.self_attn.v_proj.weight",
                    None,
                    "transpose",
                ],
                [
                    f"layers.{layer_index}.self_attn.o_proj.weight",
                    None,
                    "transpose",
                ],
                [f"layers.{layer_index}.self_attn.rotary_emb.inv_freq"],
                [f"layers.{layer_index}.mlp.gate_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.mlp.down_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.mlp.up_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.input_layernorm.weight"],
                [f"layers.{layer_index}.post_attention_layernorm.weight"],
            ]
            model_mappings.extend(layer_mappings)

        init_name_mappings(mappings=model_mappings)
        if "ErnieModelAuto" not in config.architectures:
            for mapping in model_mappings:
                mapping[0] = "model." + mapping[0]
                mapping[1] = "ernie." + mapping[1]
            model_mappings.append(["lm_head.weight", "lm_head.weight", "transpose"])

        mappings = [
            StateDictNameMapping(*mapping, index=index)
            for index, mapping in enumerate(model_mappings)
        ]
        return mappings

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):

        from paddleformers.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings(num_layers):
            final_actions = {}
            base_actions = {
                "layers.0.self_attn.q_proj.weight": partial(fn, is_column=True),
                "layers.0.self_attn.k_proj.weight": partial(fn, is_column=True),
                "layers.0.self_attn.v_proj.weight": partial(fn, is_column=True),
                "layers.0.mlp.gate_proj.weight": partial(fn, is_column=True),
                "layers.0.mlp.up_proj.weight": partial(fn, is_column=True),
                "lm_head.weight": partial(fn, is_column=not config.tie_word_embeddings),
                "embed_tokens.weight": partial(fn, is_column=False),
                "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
                "layers.0.mlp.down_proj.weight": partial(fn, is_column=False),
            }
            if config.use_bias:
                base_actions.update(
                    {
                        "layers.0.self_attn.q_proj.bias": partial(fn, is_column=True),
                        "layers.0.self_attn.k_proj.bias": partial(fn, is_column=True),
                        "layers.0.self_attn.v_proj.bias": partial(fn, is_column=True),
                        "layers.0.mlp.gate_proj.bias": partial(fn, is_column=True),
                        "layers.0.mlp.up_proj.bias": partial(fn, is_column=True),
                        "lm_head.bias": partial(fn, is_column=True),
                    }
                )
            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(
            config.num_hidden_layers
            if not config.remove_tail_layer
            else config.num_hidden_layers - 1
        )

        return mappings

    def init_weights(self, layer):
        """Initialization hook"""
        if self.config.tensor_parallel_degree > 1:
            rng_tracker = get_rng_state_tracker().rng_state
        else:
            rng_tracker = contextlib.nullcontext

        if isinstance(
            layer,
            (
                ErnieLMHead,
                nn.Embedding,
                nn.Linear,
                paddle.incubate.nn.FusedLinear,
            ),
        ):

            with rng_tracker():
                dtype = paddle.get_default_dtype()
                paddle.set_default_dtype("float32")
                if layer.weight._is_initialized():
                    if layer.weight.is_dist():
                        layer.weight._local_value().set_value(
                            paddle.randn(
                                layer.weight._local_shape, dtype=layer.weight.dtype
                            ).scale(self.config.initializer_range)
                        )
                    else:
                        layer.weight.set_value(
                            paddle.randn(
                                layer.weight.shape, dtype=layer.weight.dtype
                            ).scale(self.config.initializer_range)
                        )
                    paddle.set_default_dtype(dtype)
                    logger.info(
                        f"dist-init-fc: shape={layer.weight.shape}, "
                        f" range={self.config.initializer_range},"
                        f' type={type(layer)},norm={layer.weight.astype("float32").norm()}'
                    )

        elif isinstance(layer, RotaryEmbedding):
            head_dim = self.config.hidden_size // self.config.num_attention_heads
            inv_freq = 1.0 / (
                layer.base ** (np.arange(0, head_dim, 2).astype("float32") / head_dim)
            )

            t = np.arange(layer.max_position_embeddings, dtype="float32")
            freqs = np.einsum("i,j->ij", t, inv_freq)
            emb = np.concatenate([freqs, freqs], axis=-1)
            cos_cached = np.cos(emb)[:, :]
            sin_cached = np.sin(emb)[:, :]
            layer.cos_cached.set_value(cos_cached)
            layer.sin_cached.set_value(sin_cached)
        elif isinstance(layer, Top2Gate):
            if not hasattr(layer, "weight"):
                return
            with rng_tracker("model_parallel_rng"):
                dtype = paddle.get_default_dtype()
                paddle.set_default_dtype("float32")
                if self.config.moe_group_experts:
                    if layer.weight._is_initialized():
                        layer.weight.set_value(
                            paddle.randn(
                                layer.weight.shape, dtype=layer.weight.dtype
                            ).scale(self.config.initializer_range)
                        )
                else:
                    if layer.weight._is_initialized():
                        granularity = (
                            1
                            if self.config.moe_intermediate_size == 0
                            else self.config.intermediate_size
                            // self.config.moe_intermediate_size
                        )
                        layer.weight.set_value(
                            paddle.randn(
                                [
                                    self.config.hidden_size,
                                    self.config.moe_num_experts // granularity,
                                ],
                                dtype="float32",
                            )
                            .scale(self.config.initializer_range)
                            .repeat_interleave(granularity, axis=-1)
                        )
                logger.info(
                    f"dist-init-moe_gate: shape={layer.weight.shape}, dtype={layer.weight.dtype} "
                    f"range={self.config.initializer_range},type={type(layer)}, "
                    f'norm={layer.weight.astype("float32").norm()}'
                )


@register_base_model
class ErnieModelAuto(ErniePretrainedModelAuto):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`ErnieDecoderLayerAuto`]
    Args:
        config: ErnieMoEConfig
    """

    def __init__(self, config: ErnieMoEConfig):
        if hasattr(config, "use_moe") and config.use_moe:
            if config.moe_group in {"mp", "model", "tp", "mpdp"}:
                assert config.sequence_parallel
                logger.info(
                    f"disable FFN tensor model parallel, moe-group={config.moe_group}"
                )
                config.disable_ffn_model_parallel = True

            mesh = fleet.auto.get_mesh()
            if config.moe_group in mesh.dim_names:
                config.moe_world_size = max(1, mesh.get_dim_size(config.moe_group))
            else:
                config.moe_world_size = 1

        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.config = config

        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
        )

        if (
            self.config.tensor_parallel_degree > 1
            or self.config.pipeline_parallel_degree > 1
        ):
            self.embed_tokens.weight = dist.shard_tensor(
                self.embed_tokens.weight,
                get_mesh(),
                [dist.Replicate(), dist.Shard(1)],
            )

        layers_list = []

        def get_layer_pp_info(ipp):
            mesh = fleet.auto.get_mesh()
            if "pp" in mesh.dim_names:
                return None, False
            else:
                pp_degree = mesh.get_dim_size("pp")
                layer_num = (
                    config.num_hidden_layers - 1
                    if config.remove_tail_layer
                    else config.num_hidden_layers
                )
                layer_per_stage = math.ceil(layer_num / pp_degree)
                input_need_reshard = ipp % layer_per_stage == 0
                return ipp // layer_per_stage, input_need_reshard

        self.next_pp_stage_indexes = []
        for layer_idx in range(
            config.num_hidden_layers - 1
            if config.remove_tail_layer
            else config.num_hidden_layers
        ):
            pp_stage_id, input_need_reshard = get_layer_pp_info(layer_idx)
            layers_list.append(ErnieDecoderLayerAuto(config, layer_idx, pp_stage_id))
            if input_need_reshard:
                self.next_pp_stage_indexes.append(layer_idx)
        self.layers = nn.LayerList(layers_list)
        if config.use_rmsnorm:
            Norm = RMSNorm(config)
        elif config.use_fast_ln:
            Norm = FastLayerNorm(config)
        else:
            Norm = nn.LayerNorm(config.hidden_size, epsilon=config.rms_norm_eps)

        self.norm = Norm

        self.gradient_checkpointing = False

        self.placements = (
            [dist.Shard(1), dist.Shard(0)]
            if self.config.sequence_parallel
            else [dist.Shard(0), dist.Replicate()]
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @classmethod
    def _prepare_decoder_attention_mask(
        cls, attention_mask, input_shape, past_key_values_length, dtype
    ):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, past_key_values_length=past_key_values_length, dtype=dtype
            )

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(
                attention_mask, dtype, tgt_length=input_shape[-1]
            )
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )
        combined_attention_mask = paddle.maximum(
            combined_attention_mask.astype(dtype),
            paddle.to_tensor(float(paddle.finfo(dtype).min), dtype=dtype),
        )
        return combined_attention_mask

    def recompute_training(
        self,
        layer_module,
        hidden_states,
        attention_mask,
        position_ids,
        output_attentions,
        past_key_value,
        use_cache,
        inbatch_pack_offset,
        token_type_ids,
    ):

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs, output_gate_logits=False)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            position_ids,
            output_attentions,
            past_key_value,
            use_cache,
            inbatch_pack_offset,
            token_type_ids,
            use_reentrant=False,
        )
        return hidden_states

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
        inbatch_pack_offset=None,
        token_type_ids=None,
        **kwargs,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        seq_length_with_past = seq_length
        cache_length = 0

        if past_key_values[0] is not None:
            cache_length = paddle.shape(past_key_values[0][0])[1]
            seq_length_with_past += cache_length

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids).astype(
                self.embed_tokens.weight.dtype
            )

        global_mesh = global_mesh_starts_with_pp()
        if self.config.sequence_parallel:
            inputs_embeds = paddle.transpose(inputs_embeds, [1, 0, 2])

        if position_ids is not None:
            position_ids = dist.shard_tensor(
                position_ids,
                global_mesh,
                [dist.Replicate() for _ in range(len(global_mesh._shape))],
            )
        can_use_fa = self.config.use_flash_attn and flash_attention is not None
        can_mem_eff_attn = (
            self.config.use_mem_eff_attn and inbatch_pack_offset is not None
        )
        if can_use_fa or can_mem_eff_attn:
            if attention_mask is not None:
                attention_mask = None

        elif attention_mask is None:
            attention_mask = paddle.ones(
                (batch_size, seq_length_with_past), dtype=paddle.bool
            )

        if attention_mask is not None:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                cache_length,
                inputs_embeds.dtype,
            )
            attention_mask = dist.shard_tensor(
                attention_mask,
                global_mesh,
                [dist.Replicate() for _ in range(len(global_mesh._shape))],
            )

        hidden_states = inputs_embeds
        if self.config.tensor_parallel_degree > 1:
            hidden_states = dist.reshard(hidden_states, get_mesh(0), self.placements)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        all_router_loss = None
        if hasattr(self.config, "use_moe") and self.config.use_moe:
            all_router_loss = paddle.to_tensor(0.0)
            all_router_loss = dist.shard_tensor(
                all_router_loss, get_mesh(0), dist.Replicate()
            )
        all_gate_logits = () if hasattr(self.config, "use_moe") else None
        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            has_gradient = not hidden_states.stop_gradient
            ipp = decoder_layer.ipp
            if "pp" in fleet.auto.get_mesh().dim_names:
                if position_ids is not None:
                    position_ids_input = dist.reshard(
                        position_ids,
                        get_mesh(ipp),
                        [dist.Replicate(), dist.Replicate()],
                    )
                else:
                    position_ids_input = position_ids
                attention_mask_input = (
                    dist.reshard(
                        attention_mask,
                        get_mesh(ipp),
                        [dist.Replicate(), dist.Replicate()],
                    )
                    if attention_mask is not None
                    else None
                )
                token_type_ids_input = (
                    dist.reshard(
                        token_type_ids,
                        get_mesh(ipp),
                        [dist.Replicate(), dist.Replicate()],
                    )
                    if token_type_ids is not None
                    else None
                )
            else:
                position_ids_input = position_ids
                attention_mask_input = attention_mask
                token_type_ids_input = token_type_ids

            if idx in self.next_pp_stage_indexes:
                hidden_states = dist.reshard(
                    hidden_states,
                    get_mesh(ipp),
                    self.placements,
                )
                if hasattr(self.config, "use_moe") and self.config.use_moe:
                    all_router_loss = dist.reshard(
                        all_router_loss,
                        get_mesh(ipp),
                        [dist.Replicate()],
                    )
            if self.config.use_recompute and has_gradient:
                layer_outputs = self.recompute_training(
                    decoder_layer,
                    hidden_states,
                    attention_mask_input,
                    position_ids_input,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    inbatch_pack_offset,
                    token_type_ids_input,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask_input,
                    position_ids_input,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    inbatch_pack_offset,
                    token_type_ids_input,
                )

            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            if hasattr(self.config, "use_moe") and self.config.use_moe:
                if not (self.config.use_recompute and has_gradient):
                    layer_outputs, gate_logits = layer_outputs[:-1], layer_outputs[-1]
                    all_gate_logits = all_gate_logits + (gate_logits,)
                router_loss = layer_outputs[-1]
                all_router_loss += router_loss

        if use_cache and not (hasattr(self.config, "use_moe") and self.config.use_moe):
            hidden_states = paddle.unsqueeze(hidden_states[:, -1, :], 1)

        if self.config.pipeline_parallel_degree > 1:
            hidden_states = dist.reshard(
                hidden_states,
                get_mesh(-1),
                self.placements,
            )
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_router_loss,
                    all_gate_logits,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=None,
            router_loss=all_router_loss,
            gate_logits=all_gate_logits,
        )


class ErniePretrainingCriterionBase(paddle.nn.Layer):
    """
    Criterion for Ernie.
    It calculates the final loss.
    """

    def __init__(self, config, return_tuple=True):
        super(ErniePretrainingCriterionBase, self).__init__()
        self.ignored_index = getattr(config, "ignored_index", -100)
        self.config = config
        self.return_tuple = return_tuple
        self.enable_parallel_cross_entropy = (
            config.tensor_parallel_degree > 1 and config.tensor_parallel_output
        )

        self.loss_func = paddle.nn.CrossEntropyLoss(
            reduction="none",
        )

    def forward(self, prediction_scores, masked_lm_labels):
        if self.config.use_sparse_head_and_loss_fn:
            hidden_states, outlinear_weight, outlinear_bias = prediction_scores

            if self.config.sequence_parallel:
                masked_lm_labels, sparse_label_idx = (
                    sequence_parallel_sparse_mask_labels(
                        masked_lm_labels, self.ignored_index
                    )
                )
            else:
                masked_lm_labels = masked_lm_labels.flatten()
                sparse_label_idx = paddle.nonzero(
                    masked_lm_labels != self.ignored_index
                ).flatten()
                masked_lm_labels = paddle.take_along_axis(
                    masked_lm_labels, sparse_label_idx, axis=0
                )

                hidden_states = hidden_states.reshape([-1, hidden_states.shape[-1]])
                hidden_states = paddle.take_along_axis(
                    hidden_states, sparse_label_idx.reshape([-1, 1]), axis=0
                )

            if self.config.use_recompute_loss_fn:
                res = recompute(
                    self.forward_impl_with_calc_logits,
                    masked_lm_labels,
                    hidden_states,
                    outlinear_weight,
                    outlinear_bias,
                    sparse_label_idx,
                )
            else:
                logits = calc_lm_head_logits(
                    self.config,
                    hidden_states,
                    outlinear_weight,
                    outlinear_bias,
                    sparse_label_idx,
                )
                res = self.forward_impl(logits, masked_lm_labels)
        elif self.config.use_recompute_loss_fn:
            assert isinstance(prediction_scores, tuple) and len(prediction_scores) in [
                3,
                4,
            ]
            res = recompute(
                self.forward_impl_with_calc_logits, masked_lm_labels, *prediction_scores
            )
        else:
            res = self.forward_impl(prediction_scores, masked_lm_labels)

        return res

    def forward_impl_with_calc_logits(
        self,
        masked_lm_labels,
        hidden_states,
        outlinear_weight,
        outlinear_bias,
        sparse_label_idx=None,
        tensor_parallel_output=None,
    ):

        logits = calc_lm_head_logits(
            self.config,
            hidden_states,
            outlinear_weight,
            outlinear_bias,
            sparse_label_idx,
            tensor_parallel_output,
        )

        return self.forward_impl(logits, masked_lm_labels)

    def loss_impl(self, prediction_scores, masked_lm_labels):
        """extract loss impl for subbatch"""
        masked_lm_loss = self.loss_func(
            prediction_scores.astype("float32"), masked_lm_labels.unsqueeze(-1)
        )
        return masked_lm_loss

    def forward_impl(self, prediction_scores, masked_lm_labels):

        with paddle.amp.auto_cast(False):
            if self.config.use_sparse_head_and_loss_fn and prediction_scores.shape[
                0
            ] > self.config.get("loss_subbatch_seqlen", 32768):
                sb_loss_func = subbatch(
                    self.loss_impl,
                    [0, 1],
                    [0, 0],
                    self.config.get("loss_subbatch_seqlen", 32768),
                    0,
                )
                masked_lm_loss = sb_loss_func(prediction_scores, masked_lm_labels)
            else:
                masked_lm_loss = self.loss_impl(prediction_scores, masked_lm_labels)
            lossmask = masked_lm_labels != self.ignored_index

            if (~lossmask).all():
                logger.warning(
                    f"encounter empty span when calculate loss, ignored_index={self.ignored_index}"
                )
                loss = paddle.mean(masked_lm_loss) * 0.0
                loss_sum = masked_lm_loss.sum().detach()
            else:
                lossmask_ = lossmask.reshape([-1]).cast(paddle.float32)
                masked_lm_loss_ = paddle.sum(
                    masked_lm_loss.cast(paddle.float32).reshape([-1]) * lossmask_
                )
                loss = masked_lm_loss_ / lossmask_.sum()
                loss_sum = masked_lm_loss_.sum().detach()

        if not self.return_tuple:
            if self.training:
                return loss
            return loss_sum
        return loss, loss_sum


class ErniePretrainingCriterion(ErniePretrainingCriterionBase):
    """
    Criterion for Ernie.
    It calculates the final loss.
    """

    def __init__(self, config, return_tuple=True):
        super(ErniePretrainingCriterion, self).__init__(
            config, return_tuple=return_tuple
        )

    def forward(self, prediction_scores, masked_lm_labels, router_loss=None):
        """
        calculates the final loss
        """
        res = super().forward(
            prediction_scores,
            masked_lm_labels,
        )
        if self.return_tuple:
            loss, loss_sum = res
        else:
            loss, loss_sum = res, None
        if router_loss is not None and not in_auto_parallel_align_mode():
            global_mesh = global_mesh_starts_with_pp()
            if self.config.pipeline_parallel_degree > 1:
                loss = dist.reshard(
                    loss,
                    global_mesh,
                    [dist.Replicate() for _ in range(len(global_mesh._shape))],
                )
                router_loss = dist.reshard(
                    router_loss,
                    global_mesh,
                    [dist.Replicate() for _ in range(len(global_mesh._shape))],
                )
            loss = loss + router_loss - router_loss.detach()
        return loss, loss_sum


class ErnieLMHead(nn.Layer):
    """
    ErnieLMHead is the linear layer used to project hidden state of decoder into word embeddings.
    """

    def __init__(self, config):
        super(ErnieLMHead, self).__init__()
        self.config = config
        vocab_size = config.vocab_size
        self.weight = self.create_parameter(
            shape=(
                [vocab_size, config.hidden_size]
                if config.tie_word_embeddings
                else [config.hidden_size, vocab_size]
            ),
            dtype=paddle.get_default_dtype(),
        )

        if (
            self.config.tensor_parallel_degree > 1
            or self.config.pipeline_parallel_degree > 1
        ):
            self.weight = dist.shard_tensor(
                self.weight,
                get_mesh(-1),
                [dist.Replicate(), dist.Shard(1)],
            )

        logger.info(
            f"output-weight:{self.weight.shape} config.tie_word_embeddings={config.tie_word_embeddings}"
        )
        if config.weight_share_add_bias and config.use_bias:
            self.bias = self.create_parameter(
                shape=[vocab_size],
                dtype=paddle.get_default_dtype(),
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.constant.Constant(0.0)
                ),
            )
            if (
                self.config.tensor_parallel_degree > 1
                or self.config.pipeline_parallel_degree > 1
            ):
                self.bias = dist.shard_tensor(
                    self.bias,
                    get_mesh(-1),
                    [dist.Replicate(), dist.Shard(0)],
                )
        else:
            self.bias = None

        self.weight.is_distributed = (
            True if (vocab_size != config.vocab_size) else False
        )
        if config.weight_share_add_bias and config.use_bias:
            self.bias.is_distributed = (
                True if (vocab_size != config.vocab_size) else False
            )

        if self.weight.is_distributed:
            self.weight.split_axis = 1
        if (
            config.weight_share_add_bias
            and config.use_bias
            and self.bias.is_distributed
        ):
            self.bias.split_axis = 0

        if self.config.use_recompute_loss_fn:
            logger.info(
                "Using recompute_loss_fn, the calculation of logits will be moved into "
                "loss_fn for memory optimization"
            )

    def forward(self, hidden_states, tensor_parallel_output=None):

        if self.config.use_recompute_loss_fn or self.config.use_sparse_head_and_loss_fn:
            out_tensors = (
                (hidden_states, self.weight, self.bias)
                if tensor_parallel_output is None
                else (hidden_states, self.weight, self.bias, tensor_parallel_output)
            )

            return out_tensors

        return calc_lm_head_logits(
            self.config,
            hidden_states,
            self.weight,
            self.bias,
            None,
            tensor_parallel_output,
        )


class ErnieForCausalLMAuto(ErniePretrainedModelAuto):
    """
    ErnieForCausalLMAuto is the model class for causal language modeling.
    """

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)

        if config.sequence_parallel:
            logger.info(f"using sequence_parallel, input seqlen={config.seqlen}")
            if config.using_dynamic_sequence_length:
                assert (
                    not config.micro_batch_size
                ), "sequence-parallel needs micro_batch_size setting when using dygramic_sequnence_length"
            else:
                assert config.seqlen is not None

            assert (
                config.tensor_parallel_degree > 1
            ), f"sequence-parallel needs mp>1, got mp={config.tensor_parallel_degree}"

        config.initializer_range = math.sqrt(0.3333 / config.hidden_size)
        self.config = config
        self.ernie = ErnieModelAuto(config)
        self.lm_head = ErnieLMHead(config)
        self.criterion = ErniePretrainingCriterion(config)

        self.tie_weights()

    def _post_init(self, original_init, *args, **kwargs):
        """
        Initialize weights and apply final processing
        """
        super()._post_init(self, original_init, *args, **kwargs)
        factor = 1 / math.sqrt(2 * self.config.num_hidden_layers)
        logger.info(f"using post init div: factor:{factor}")

        def scale_by_factor_if_valid(w):
            if w.is_dist() and w._is_initialized():
                w.scale_(factor)

        if hasattr(self.config, "use_moe") and self.config.use_moe:
            with paddle.no_grad():
                for left in self.ernie.layers:
                    if isinstance(
                        left.self_attn.o_proj,
                        (MOELayerAuto),
                    ):
                        for e in left.self_attn.o_proj.experts:
                            if isinstance(e, ErnieMoeMLP):
                                scale_by_factor_if_valid(e.weight)
                    else:
                        scale_by_factor_if_valid(left.self_attn.o_proj.weight)

                    if isinstance(
                        left.mlp,
                        (MOELayerAuto),
                    ):
                        for e in left.mlp.experts:
                            if isinstance(e, ErnieMoeMLP):
                                scale_by_factor_if_valid(e.down_proj.weight)
                    else:
                        scale_by_factor_if_valid(left.mlp.down_proj.weight)
        else:
            with paddle.no_grad():
                for left in self.ernie.layers:
                    scale_by_factor_if_valid(left.self_attn.o_proj.weight)
                    scale_by_factor_if_valid(left.mlp.down_proj.weight)

    def get_input_embeddings(self):

        return self.ernie.embed_tokens

    def set_input_embeddings(self, value):

        self.ernie.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.ernie = decoder

    def get_decoder(self):
        return self.ernie

    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id):
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(
            input_ids == pad_token_id
        ).numpy().item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            attention_mask = (input_ids != pad_token_id).astype("int64")
        else:
            attention_mask = paddle.ones_like(input_ids, dtype="int64")
        return attention_mask

    def prepare_inputs_for_generation(
        self,
        input_ids,
        use_cache=False,
        past_key_values=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        attention_mask = kwargs.get("attention_mask", None)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": True,
                "attention_mask": attention_mask,
                "return_dict": True,
            }
        )
        return model_inputs

    @staticmethod
    def update_model_kwargs_for_generation(
        outputs, model_kwargs, is_encoder_decoder=False
    ):
        if (
            isinstance(outputs, tuple)
            and len(outputs) > 1
            and not isinstance(outputs[1], paddle.Tensor)
        ):
            model_kwargs["past_key_values"] = outputs[1]

        if (
            isinstance(outputs, CausalLMOutputWithCrossAttentions)
            and "past_key_values" in outputs
        ):
            model_kwargs["past_key_values"] = outputs.past_key_values

        if (
            "token_type_ids" in model_kwargs
            and model_kwargs["token_type_ids"] is not None
        ):
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.concat(
                [token_type_ids, token_type_ids[:, -1:]], axis=-1
            )

        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = paddle.concat(
                    [
                        attention_mask,
                        paddle.ones([attention_mask.shape[0], 1], dtype="int64"),
                    ],
                    axis=-1,
                )
        if "role_ids" in model_kwargs and model_kwargs["role_ids"] is not None:
            role_ids = model_kwargs["role_ids"]
            model_kwargs["role_ids"] = paddle.concat(
                [role_ids, role_ids[:, -1:]], axis=-1
            )

        return model_kwargs

    def forward(
        self,
        input_ids,
        labels=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=False,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=False,
        ignored_index=0,
        inbatch_pack_offset=None,
        token_type_ids=None,
    ):
        if isinstance(input_ids, list):
            input_ids, labels = input_ids[:2]

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.ernie(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            inbatch_pack_offset=inbatch_pack_offset,
            token_type_ids=token_type_ids,
        )

        hidden_states = outputs.last_hidden_state

        logits = self.lm_head(
            hidden_states,
        )

        if return_dict:
            if labels is not None:
                loss, _ = self.criterion(logits, labels)
            else:
                loss = None
            return CausalLMOutputWithCrossAttentionsAuto(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                router_loss=outputs.router_loss if self.config.use_moe else None,
            )

        assert labels is not None
        router_loss = (
            outputs.router_loss
            if hasattr(self.config, "use_moe") and self.config.use_moe
            else None
        )
        return self.criterion(logits, labels, router_loss)
