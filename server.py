"""Stores group GANs. Scheduling, clients, mappings and evaluation live elsewhere."""
from contracts import AggregationStrategy, ClientUpdate, GANState, clone_state


class Server:
    def __init__(self, aggregation: AggregationStrategy):
        self.aggregation = aggregation
        self._groups: dict[str, GANState] = {}

    def aggregate(self, messages: list[ClientUpdate]) -> None:
        # Compute first: a failed strategy cannot partially overwrite stored groups.
        result = self.aggregation(messages)
        for group, state in result.items():
            self._groups[group] = GANState(clone_state(state.generator),
                                           clone_state(state.discriminator))

    def snapshot(self) -> dict[str, GANState]:
        return {group: GANState(clone_state(s.generator), clone_state(s.discriminator))
                for group, s in self._groups.items()}

