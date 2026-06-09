from typing import Optional

import itertools
import torch


def get_vision_position_ids(
    start_position: int,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    device: torch.device,
) -> torch.Tensor:
    llm_grid_t, llm_grid_h, llm_grid_w = (
        grid_thw[0].item(),
        grid_thw[1].item() // spatial_merge_size,
        grid_thw[2].item() // spatial_merge_size,
    )

    position_temporal = torch.arange(llm_grid_t, device=device)
    position_width = torch.arange(llm_grid_w, device=device) + start_position
    position_height = torch.arange(llm_grid_h, device=device) + start_position

    position_width = position_width.repeat(llm_grid_h * llm_grid_t)
    position_height = position_height.repeat_interleave(llm_grid_w).repeat(llm_grid_t)
    position_temporal = position_temporal.repeat_interleave(llm_grid_h * llm_grid_w) + start_position
    return torch.stack([position_temporal, position_height, position_width], dim=0)


def build_mm_token_type_ids(
    input_ids: torch.Tensor,
    image_token_id: int,
    video_token_id: int,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mm_token_type_ids = torch.zeros_like(input_ids)
    valid_mask = attention_mask.bool() if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)
    mm_token_type_ids[(input_ids == image_token_id) & valid_mask] = 1
    mm_token_type_ids[(input_ids == video_token_id) & valid_mask] = 2
    return mm_token_type_ids


def get_rope_index(
    processor,
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    mm_token_type_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build Qwen3-VL 4-channel position ids: text, temporal, height, width.
    This mirrors the transformers Qwen3VL implementation while accepting the
    single-example tensors produced by this dataset.
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    if mm_token_type_ids is None:
        mm_token_type_ids = build_mm_token_type_ids(input_ids, image_token_id, video_token_id, attention_mask)

    spatial_merge_size = getattr(processor.video_processor, "merge_size", None)
    if spatial_merge_size is None:
        spatial_merge_size = getattr(processor.image_processor, "merge_size", 2)

    expanded_video_grid_thw = None
    if video_grid_thw is not None:
        expanded_video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        expanded_video_grid_thw[:, 0] = 1

    grid_iters = {
        1: iter(image_grid_thw) if image_grid_thw is not None else None,
        2: iter(expanded_video_grid_thw) if expanded_video_grid_thw is not None else None,
    }

    position_ids_3d = torch.zeros(3, input_ids.shape[0], dtype=input_ids.dtype, device=input_ids.device)
    valid_token_type = mm_token_type_ids[attention_mask.bool()]

    input_type_group = []
    for key, group in itertools.groupby(enumerate(valid_token_type.tolist()), lambda x: x[1]):
        group = list(group)
        input_type_group.append((key, group[0][0], group[-1][0] + 1))

    current_pos = 0
    llm_pos_ids_list = []
    for modality_type, start_idx, end_idx in input_type_group:
        if modality_type == 0:
            text_len = end_idx - start_idx
            llm_pos_ids_list.append(
                torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
            )
            current_pos += text_len
        else:
            if grid_iters[modality_type] is None:
                raise ValueError(f"Missing grid_thw for Qwen3-VL modality type {modality_type}")
            grid_thw = next(grid_iters[modality_type])
            vision_position_ids = get_vision_position_ids(
                current_pos,
                grid_thw,
                spatial_merge_size=spatial_merge_size,
                device=input_ids.device,
            )
            llm_pos_ids_list.append(vision_position_ids)
            current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size

    llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    position_ids_3d[:, attention_mask.bool()] = llm_positions.to(position_ids_3d.device)

    text_position_ids = attention_mask.long().cumsum(-1) - 1
    text_position_ids.masked_fill_(attention_mask == 0, 1)
    return torch.cat([text_position_ids.unsqueeze(0), position_ids_3d], dim=0)
