# -*- coding: utf-8 -*-

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


def _to_int_list(record_len: Union[Sequence[int], torch.Tensor]) -> List[int]:
    if isinstance(record_len, torch.Tensor):
        return [int(x) for x in record_len.view(-1).tolist()]
    return [int(x) for x in record_len]


def _normalize_sample_indices(
    sample_indices: Optional[Union[int, Sequence[int], torch.Tensor]],
    batch_size: int,
) -> List[int]:
    if sample_indices is None:
        return list(range(batch_size))
    if isinstance(sample_indices, torch.Tensor):
        values = [int(x) for x in sample_indices.view(-1).tolist()]
    elif isinstance(sample_indices, (list, tuple)):
        values = [int(x) for x in sample_indices]
    else:
        values = [int(sample_indices)]

    if len(values) == 1 and batch_size > 1:
        values = values * batch_size
    if len(values) != batch_size:
        raise ValueError(
            f"sample_indices length mismatch: expected {batch_size}, got {len(values)}"
        )
    return values


def _compose_seed(seed_base: int, sample_idx: int, agent_idx: int) -> int:
    return int(seed_base) + int(sample_idx) * 100_003 + int(agent_idx) * 1_000_003


def _sample_uniform(
    shape: Tuple[int, ...],
    device: torch.device,
    seed: Optional[int],
) -> torch.Tensor:
    if seed is None:
        return torch.rand(shape, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.rand(shape, generator=generator).to(device=device)


def build_spatial_packet_loss_mask(
    record_len: Union[Sequence[int], torch.Tensor],
    spatial_size: Tuple[int, int],
    keep_ratio: float,
    device: torch.device,
    dtype: torch.dtype,
    mode: str = "bernoulli",
    burst_coarse_shape: Tuple[int, int] = (8, 16),
    sample_indices: Optional[Union[int, Sequence[int], torch.Tensor]] = None,
    seed_base: Optional[int] = None,
) -> torch.Tensor:
    batch_record_len = _to_int_list(record_len)
    total_agents = sum(batch_record_len)
    height, width = int(spatial_size[0]), int(spatial_size[1])

    if not (0.0 <= keep_ratio <= 1.0):
        raise ValueError(f"keep_ratio must be in [0, 1], got {keep_ratio}")

    if keep_ratio >= 1.0:
        return torch.ones((total_agents, 1, height, width), device=device, dtype=dtype)

    if keep_ratio <= 0.0:
        mask = torch.zeros((total_agents, 1, height, width), device=device, dtype=dtype)
        start_idx = 0
        for num_agents in batch_record_len:
            mask[start_idx] = 1.0
            start_idx += num_agents
        return mask

    normalized_mode = mode.lower()
    if normalized_mode not in {"bernoulli", "burst"}:
        raise ValueError(f"Unsupported packet loss mode: {mode}")

    coarse_h = max(1, min(int(burst_coarse_shape[0]), height))
    coarse_w = max(1, min(int(burst_coarse_shape[1]), width))
    batch_sample_indices = _normalize_sample_indices(sample_indices, len(batch_record_len))

    mask = torch.ones((total_agents, 1, height, width), device=device, dtype=dtype)
    start_idx = 0

    for batch_idx, num_agents in enumerate(batch_record_len):
        sample_idx = batch_sample_indices[batch_idx]
        for local_agent_idx in range(num_agents):
            global_agent_idx = start_idx + local_agent_idx
            if local_agent_idx == 0:
                mask[global_agent_idx] = 1.0
                continue

            seed = None
            if seed_base is not None:
                seed = _compose_seed(int(seed_base), sample_idx, local_agent_idx)

            if normalized_mode == "bernoulli":
                random_map = _sample_uniform((1, 1, height, width), device=device, seed=seed)
                agent_mask = (random_map < keep_ratio).to(dtype=dtype)
            else:
                coarse_random = _sample_uniform((1, 1, coarse_h, coarse_w), device=device, seed=seed)
                coarse_mask = (coarse_random < keep_ratio).to(dtype=torch.float32)
                agent_mask = F.interpolate(
                    coarse_mask,
                    size=(height, width),
                    mode="nearest",
                ).to(dtype=dtype)

            mask[global_agent_idx] = agent_mask[0]

        start_idx += num_agents

    return mask
