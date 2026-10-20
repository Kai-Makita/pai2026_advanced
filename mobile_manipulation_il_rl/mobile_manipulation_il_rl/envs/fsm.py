from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Phase(IntEnum):
    NAVIGATE_TO_PICK = 0
    REACH = 1
    GRASP = 2
    LIFT = 3
    NAVIGATE_TO_PLACE = 4
    PLACE = 5
    SUCCESS = 6
    FAILURE = 7


OBSERVABLE_PHASES = tuple(Phase)[:6]


@dataclass
class Transition:
    changed: bool = False
    reason: str = ""


class TaskFSM:
    def __init__(self, timeouts: dict[str, float]):
        self.timeouts = timeouts
        self.reset()

    def reset(self) -> None:
        self.phase = Phase.NAVIGATE_TO_PICK
        self.phase_time = 0.0
        self.failure_reason = ""

    def tick(self, dt: float) -> Transition:
        self.phase_time += dt
        if self.phase.name in self.timeouts and self.phase_time > self.timeouts[self.phase.name]:
            return self.fail(f"timeout:{self.phase.name}")
        return Transition()

    def advance(self, reason: str) -> Transition:
        if self.phase.value >= Phase.PLACE.value:
            self.phase = Phase.SUCCESS
        else:
            self.phase = Phase(self.phase.value + 1)
        self.phase_time = 0.0
        return Transition(True, reason)

    def fail(self, reason: str) -> Transition:
        self.phase = Phase.FAILURE
        self.failure_reason = reason
        return Transition(True, reason)

    @property
    def terminal(self) -> bool:
        return self.phase in (Phase.SUCCESS, Phase.FAILURE)

