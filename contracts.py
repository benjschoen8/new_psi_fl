"""Messages exchanged by the orchestrator; none contains a participant object."""
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol

State = dict[str, Any]
RelationTable = dict[str, dict[int, int]]


@dataclass(frozen=True)
class GANState:
    generator: State
    discriminator: State


@dataclass(frozen=True)
class ClientUpdate:
    client_id: int
    group: str
    num_samples: int
    gan: GANState


@dataclass(frozen=True)
class MappingInputs:
    generators: Mapping[str, Any]
    classifiers: Mapping[str, list[Any]]
    num_classes: Mapping[str, int]
    # Each group owner's own ordered class names; PSI strategies reveal them only via PSI.
    label_names: Optional[Mapping[str, tuple]] = None


class MappingStrategy(Protocol):
    def __call__(self, inputs: MappingInputs) -> RelationTable: ...


class ClusteringStrategy(Protocol):
    def __call__(self, bases: Mapping[int, Any]) -> dict[int, str]: ...


class AggregationStrategy(Protocol):
    def __call__(self, updates: list[ClientUpdate]) -> dict[str, GANState]: ...


def clone_state(state: Mapping[str, Any]) -> State:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def payload_bytes(state: GANState) -> int:
    return sum(t.numel() * t.element_size()
               for values in (state.generator, state.discriminator) for t in values.values())
