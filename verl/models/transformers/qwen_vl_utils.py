import torch


def is_qwen3_vl_processor(processor) -> bool:
    return processor is not None and processor.__class__.__name__ == "Qwen3VLProcessor"


def video_processor_kwargs(processor) -> dict:
    return {"return_metadata": True} if is_qwen3_vl_processor(processor) else {}


def get_vl_rope_index(processor, **kwargs):
    if is_qwen3_vl_processor(processor):
        from verl.models.transformers.qwen3_vl import get_rope_index
    else:
        from verl.models.transformers.qwen2_vl import get_rope_index
    return get_rope_index(processor, **kwargs)


def get_text_position_ids(position_ids):
    if position_ids.dim() == 3:
        return position_ids[:, 0, :]
    return position_ids


def append_response_position_ids(position_ids, response_length: int, batch_size: int):
    delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device, dtype=position_ids.dtype)
    delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
    if position_ids.dim() == 3:
        delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)
        response_position_ids = position_ids[:, :, -1:] + delta_position_id
    else:
        response_position_ids = position_ids[:, -1:] + delta_position_id
    return torch.cat([position_ids, response_position_ids], dim=-1)
