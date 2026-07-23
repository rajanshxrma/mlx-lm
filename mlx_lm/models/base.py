# Copyright © 2023-2024 Apple Inc.

import inspect
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
from mlx.utils import tree_map


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def create_causal_mask(
    N: int,
    offset: int = 0,
    window_size: Optional[int] = None,
    right_padding: Optional[mx.array] = None,
    left_padding: Optional[mx.array] = None,
):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    if right_padding is not None:
        mask = mask & (rinds < mx.expand_dims((offset + N) - right_padding, (1, 2, 3)))
    if left_padding is not None:
        mask = mask & (mx.expand_dims(left_padding, (1, 2, 3)) <= rinds)
    return mask


def create_attention_mask(
    h, cache=None, window_size: Optional[int] = None, return_array: bool = False
):
    N = h.shape[1]
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(N, return_array=return_array, window_size=window_size)
    if N == 1:
        return None
    if return_array or (window_size and N > window_size):
        return create_causal_mask(N, window_size=window_size)
    return "causal"


def create_ssm_mask(h, cache=None):
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(h.shape[1])
    return None


def quantized_scaled_dot_product_attention(
    queries: mx.array,
    q_keys: tuple[mx.array, mx.array, mx.array],
    q_values: tuple[mx.array, mx.array, mx.array],
    scale: float,
    mask: Optional[mx.array],
    group_size: int = 64,
    bits: int = 8,
) -> mx.array:
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    queries *= scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    scores = mx.quantized_matmul(
        queries, *q_keys, transpose=True, group_size=group_size, bits=bits
    )
    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
        if n_repeats > 1 and mask.ndim > 3:
            mask = mx.expand_dims(mask, -3)
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores += mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(
        scores, *q_values, transpose=False, group_size=group_size, bits=bits
    )

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))

    return out


def quantized_scaled_dot_product_attention_blockwise_reference(
    queries: mx.array,
    q_keys: tuple[mx.array, mx.array, mx.array],
    q_values: tuple[mx.array, mx.array, mx.array],
    scale: float,
    mask,
    group_size: int = 64,
    bits: int = 8,
    kv_block_size: int = 512,
) -> mx.array:
    """
    Reference-only, NOT wired into `scaled_dot_product_attention` and NOT
    intended as a merge candidate. Kept alongside `quantized_scaled_dot_product_attention`
    as documentation of a confirmed mechanism and a starting point for whoever
    picks up a real fix; see https://github.com/ml-explore/mlx-lm/issues/1587
    and https://github.com/rajanshxrma/mlx-kvcache-1587 for the full
    investigation, raw data, and correctness/memory measurements.

    `quantized_scaled_dot_product_attention` above materializes the entire
    `(B, n_kv_heads, n_repeats, L, context)` score matrix in one call, with no
    tiling over the key/context dimension -- unlike the fp16 path's fused
    `mx.fast.scaled_dot_product_attention`. For long-context prefill this is a
    large, real, and unnecessary peak-memory transient (confirmed via chunk-size
    scaling: peak memory scales with the prefill chunk size at fixed context,
    while the fp16 control does not -- see the investigation repo).

    This function tiles over the key/context dimension with an online (running)
    softmax -- the standard flash-attention approach -- so only a
    `(..., L, kv_block_size)` score buffer is ever materialized, independent of
    total context length. It is numerically verified against
    `quantized_scaled_dot_product_attention` above (max abs diff <= 1e-6 across
    causal/none/boolean/additive mask forms and both non-GQA and GQA configs;
    see the investigation repo's `correctness_check_output.txt`).

    It is NOT proposed as a drop-in replacement, for one important reason found
    during that investigation: naive Python-level tiling is *worse* than the
    current implementation under MLX's lazy evaluation, which keeps every
    block's intermediate graph nodes alive until the final eval rather than
    freeing each block before the next starts -- so a pure-Python loop like
    this one pays the *sum* of all blocks' buffers, not the peak of any one
    block. Forcing evaluation after each block restores the expected
    memory-bounding behavior (and drops prefill peak below the fp16 baseline at
    long context in testing), but costs roughly 2x prefill latency from
    per-block dispatch overhead in pure Python. The real fix this points to is
    a fused kernel for quantized attention (the quantized analog of
    `mx.fast.scaled_dot_product_attention`), which would tile internally
    without either the lazy-eval memory blowup or the Python dispatch cost --
    a C++/Metal-level change, not a Python-level one. This function exists so
    the mechanism and the correctness of the tiling approach are on record and
    reproducible, not as something to call from `scaled_dot_product_attention`.
    """
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    # Non-mutating, unlike quantized_scaled_dot_product_attention's `queries *=
    # scale` in place -- avoids surprising a caller that holds a reference to
    # the original `queries` array (relevant for the correctness check above,
    # which runs both implementations against the same input).
    queries = queries * scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    kL = q_keys[0].shape[-2]

    is_causal_str = isinstance(mask, str)
    mask_arr = None
    if mask is not None and not is_causal_str:
        mask_arr = mask
        if n_repeats > 1 and mask_arr.ndim > 3:
            mask_arr = mx.expand_dims(mask_arr, -3)

    lead_shape = queries.shape[:-1]
    m = mx.full(lead_shape + (1,), -mx.inf, dtype=mx.float32)
    l = mx.zeros(lead_shape + (1,), dtype=mx.float32)
    acc = mx.zeros(queries.shape[:-1] + (D,), dtype=mx.float32)

    import math

    n_blocks = math.ceil(kL / kv_block_size)
    for bi in range(n_blocks):
        blk_start = bi * kv_block_size
        blk_end = min(blk_start + kv_block_size, kL)

        keys_blk = tree_map(lambda x: x[..., blk_start:blk_end, :], q_keys)
        values_blk = tree_map(lambda x: x[..., blk_start:blk_end, :], q_values)

        block_scores = mx.quantized_matmul(
            queries, *keys_blk, transpose=True, group_size=group_size, bits=bits
        )
        block_scores = block_scores.astype(mx.float32)

        if is_causal_str:
            q_indices = mx.arange(kL - L, kL)
            k_indices = mx.arange(blk_start, blk_end)
            causal_blk = q_indices[:, None] >= k_indices[None]
            block_scores = mx.where(
                causal_blk, block_scores, mx.array(-mx.inf, dtype=mx.float32)
            )
        elif mask_arr is not None:
            mblk = mask_arr[..., :, blk_start:blk_end]
            if mask_arr.dtype == mx.bool_:
                block_scores = mx.where(
                    mblk, block_scores, mx.array(-mx.inf, dtype=mx.float32)
                )
            else:
                block_scores = block_scores + mblk.astype(mx.float32)

        block_max = mx.max(block_scores, axis=-1, keepdims=True)
        new_m = mx.maximum(m, block_max)

        # Guard against an all-masked block turning -inf - -inf into nan.
        finite_new_m = mx.where(mx.isinf(new_m), mx.zeros_like(new_m), new_m)
        correction = mx.where(
            mx.isinf(new_m), mx.zeros_like(m), mx.exp(m - finite_new_m)
        )
        p = mx.where(
            mx.isinf(block_scores),
            mx.zeros_like(block_scores),
            mx.exp(block_scores - finite_new_m),
        )

        l = l * correction + mx.sum(p, axis=-1, keepdims=True)
        block_out = mx.quantized_matmul(
            p.astype(queries.dtype), *values_blk,
            transpose=False, group_size=group_size, bits=bits,
        )
        acc = acc * correction + block_out.astype(mx.float32)
        m = new_m

    safe_l = mx.where(l == 0, mx.ones_like(l), l)
    out = acc / safe_l
    out = out.astype(queries.dtype)

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))

    return out


def scaled_dot_product_attention(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: Optional[mx.array],
    sinks: Optional[mx.array] = None,
) -> mx.array:
    if hasattr(cache, "bits"):
        if sinks is not None:
            raise ValueError("Quantized SDPA does not support attention sinks.")
        return quantized_scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            group_size=cache.group_size,
            bits=cache.bits,
        )
    else:
        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            sinks=sinks,
        )
