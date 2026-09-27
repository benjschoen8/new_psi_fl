"""Sample-weighted GeFL aggregation of GANs within PACFL groups."""
from collections import defaultdict
import torch
from contracts import ClientUpdate, GANState


def weighted_average(weighted_states):
    if not weighted_states or any(n <= 0 for n, _ in weighted_states):
        raise ValueError('Aggregation requires positive sample counts')
    reference = weighted_states[0][1]
    total = sum(n for n, _ in weighted_states)
    for _, state in weighted_states:
        if state.keys() != reference.keys():
            raise ValueError('Incompatible model state keys within a group')
        if any(state[k].shape != reference[k].shape or state[k].dtype != reference[k].dtype
               for k in reference):
            raise ValueError('Incompatible model shapes/dtypes within a group')
    result = {}
    for key, value in reference.items():
        # Legacy FedAvg accumulated float32, then load_state_dict cast buffers back.
        averaged = torch.zeros_like(value, device='cpu', dtype=torch.float32)
        for count, state in weighted_states:
            averaged.add_(state[key].detach().cpu().float(), alpha=count / total)
        result[key] = averaged.to(value.dtype)
    return result


class GeFLAggregation:
    def __call__(self, updates: list[ClientUpdate]) -> dict[str, GANState]:
        grouped = defaultdict(list)
        seen = set()
        for message in updates:
            if message.client_id in seen:
                raise ValueError('Duplicate client update')
            seen.add(message.client_id)
            grouped[message.group].append(message)
        return {group: GANState(
            weighted_average([(m.num_samples, m.gan.generator) for m in messages]),
            weighted_average([(m.num_samples, m.gan.discriminator) for m in messages]),
        ) for group, messages in grouped.items()}

