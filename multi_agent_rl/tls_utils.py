import traci
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class TrafficLightPhase:
    phase_index: int
    duration: int
    state: str  # e.g., "GrGr", "yryr", etc.


@dataclass
class TrafficLightInfo:
    tls_id: str
    green_phases: List[int]  # Indices of green phases to be modified
    fixed_yellow: int = 3  # seconds


def get_tls_info() -> Dict[str, TrafficLightInfo]:
    """
    Extracts traffic light information for each TLS in the SUMO network using TraCI.
    All phases except yellow phases (those containing 'y') are considered modifiable.
    """
    tls_infos = {}
    traci.start(["sumo", "-c", "osm.sumocfg", "--no-step-log", "true"])

    for tls_id in traci.trafficlight.getIDList():
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
        green_phases = []

        for i, phase in enumerate(logic.phases):
            if "y" in phase.state:
                continue  # Skip yellow phases
            green_phases.append(i)

        tls_infos[tls_id] = TrafficLightInfo(
            tls_id=tls_id,
            green_phases=green_phases,
            fixed_yellow=3
        )

    traci.close()
    return tls_infos

