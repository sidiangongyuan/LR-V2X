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


def contiguous_numeric_sequence_ends(
    frame_ids: Sequence[Union[int, str]],
) -> List[int]:
    """Return cumulative ends of contiguous numeric frame-ID runs."""
    if not frame_ids:
        return []

    numeric_ids = [int(frame_id) for frame_id in frame_ids]
    sequence_ends = []
    for index in range(1, len(numeric_ids)):
        if numeric_ids[index] != numeric_ids[index - 1] + 1:
            sequence_ends.append(index)
    sequence_ends.append(len(numeric_ids))
    return sequence_ends


def grouped_sequence_ends(sequence_ids: Sequence[Union[int, str]]) -> List[int]:
    """Return cumulative ends for adjacent samples sharing one sequence ID."""
    if not sequence_ids:
        return []

    normalized_ids = [str(sequence_id) for sequence_id in sequence_ids]
    sequence_ends = []
    for index in range(1, len(normalized_ids)):
        if normalized_ids[index] != normalized_ids[index - 1]:
            sequence_ends.append(index)
    sequence_ends.append(len(normalized_ids))
    return sequence_ends


def encode_temporal_sample_index(
    dataset_index: int,
    sequence_end_indices: Optional[Sequence[int]],
    temporal_block_len: int,
) -> int:
    """Encode a sequence-local temporal block for the existing mask API."""
    block_len = int(temporal_block_len)
    if block_len < 1:
        raise ValueError(f"temporal_block_len must be positive, got {block_len}")
    index = int(dataset_index)
    if not sequence_end_indices:
        return index

    sequence_idx = 0
    sequence_start = 0
    for sequence_end in sequence_end_indices:
        if index < int(sequence_end):
            break
        sequence_start = int(sequence_end)
        sequence_idx += 1
    local_block_idx = (index - sequence_start) // block_len
    composite_block_idx = sequence_idx * 1_000_003 + local_block_idx
    return composite_block_idx * block_len


def encode_packet_loss_sample_index(
    dataset_index: int,
    mode: str,
    sequence_end_indices: Optional[Sequence[int]],
    temporal_block_len: int,
) -> int:
    """Return a deterministic per-sample index for the selected loss process."""
    if mode.lower() == "temporal_block":
        return encode_temporal_sample_index(
            dataset_index,
            sequence_end_indices,
            temporal_block_len,
        )
    return int(dataset_index)


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
    temporal_block_len: int = 1,
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
    if normalized_mode not in {"bernoulli", "burst", "temporal_block"}:
        raise ValueError(f"Unsupported packet loss mode: {mode}")
    if int(temporal_block_len) < 1:
        raise ValueError(
            f"temporal_block_len must be positive, got {temporal_block_len}"
        )
    if normalized_mode == "temporal_block" and seed_base is None:
        raise ValueError("temporal_block mode requires seed_base for reproducibility")

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
                seed_sample_idx = sample_idx
                if normalized_mode == "temporal_block":
                    seed_sample_idx = sample_idx // int(temporal_block_len)
                seed = _compose_seed(
                    int(seed_base),
                    seed_sample_idx,
                    local_agent_idx,
                )

            if normalized_mode in {"bernoulli", "temporal_block"}:
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
