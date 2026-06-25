"""SHIFT equipment helpers for the Marshfield Baseline Network."""

from __future__ import annotations

import math
from functools import cached_property

import numpy as np
from gdm.distribution import DistributionSystem
from gdm.distribution.components import DistributionBranchBase
from gdm.distribution.components import DistributionLoad
from gdm.distribution.components import DistributionTransformer
from gdm.distribution.components import DistributionVoltageSource
from gdm.distribution.enums import ConnectionType
from gdm.distribution.enums import Phase
from gdm.distribution.enums import VoltageTypes
from gdm.distribution.equipment import DistributionTransformerEquipment
from gdm.distribution.equipment import LoadEquipment
from gdm.distribution.equipment import MatrixImpedanceBranchEquipment
from gdm.distribution.equipment import PhaseLoadEquipment
from gdm.distribution.equipment import PhaseVoltageSourceEquipment
from gdm.distribution.equipment import VoltageSourceEquipment
from gdm.distribution.equipment import WindingEquipment
from gdm.quantities import ApparentPower
from gdm.quantities import Current
from gdm.quantities import Reactance
from gdm.quantities import ReactivePower
from gdm.quantities import Voltage
from infrasys.quantities import ActivePower
from infrasys.quantities import Angle
from infrasys.quantities import Resistance
from shift import EdgeEquipmentMapper


def equipment_catalog(prefix: str = "example") -> DistributionSystem:
    """Build a local equipment catalog compatible with current SHIFT/GDM objects."""
    catalog = DistributionSystem(name=f"{prefix}_equipment_catalog")

    for phase_count in (1, 2, 3):
        catalog.add_component(
            MatrixImpedanceBranchEquipment(
                name=f"{prefix}_branch_{phase_count}ph",
                r_matrix=np.eye(phase_count) * 0.4013,
                x_matrix=np.eye(phase_count) * 0.2809,
                c_matrix=np.zeros((phase_count, phase_count)),
                ampacity=Current(10000, "ampere"),
            )
        )

    split_phase_windings = [
        WindingEquipment(
            name=f"{prefix}_split_primary",
            num_phases=1,
            rated_power=ApparentPower(5000, "kilovolt_ampere"),
            rated_voltage=Voltage(7.2, "kilovolt"),
            voltage_type=VoltageTypes.LINE_TO_GROUND,
            connection_type=ConnectionType.STAR,
            resistance=0.6,
            is_grounded=True,
            tap_positions=[1.0],
        ),
        WindingEquipment(
            name=f"{prefix}_split_secondary_s1",
            num_phases=1,
            rated_power=ApparentPower(5000, "kilovolt_ampere"),
            rated_voltage=Voltage(120, "volt"),
            voltage_type=VoltageTypes.LINE_TO_GROUND,
            connection_type=ConnectionType.STAR,
            resistance=0.012,
            is_grounded=True,
            tap_positions=[1.0],
        ),
        WindingEquipment(
            name=f"{prefix}_split_secondary_s2",
            num_phases=1,
            rated_power=ApparentPower(5000, "kilovolt_ampere"),
            rated_voltage=Voltage(120, "volt"),
            voltage_type=VoltageTypes.LINE_TO_GROUND,
            connection_type=ConnectionType.STAR,
            resistance=0.012,
            is_grounded=True,
            tap_positions=[1.0],
        ),
    ]
    catalog.add_components(*split_phase_windings)
    catalog.add_component(
        DistributionTransformerEquipment(
            name=f"{prefix}_split_phase_transformer",
            windings=split_phase_windings,
            is_center_tapped=True,
            pct_no_load_loss=0.1,
            pct_full_load_loss=1.0,
            coupling_sequences=[[0, 1], [0, 2], [1, 2]],
            winding_reactances=[0.02, 0.02, 0.01],
        )
    )

    three_phase_windings = [
        WindingEquipment(
            name=f"{prefix}_three_phase_primary",
            num_phases=3,
            rated_power=ApparentPower(5000, "kilovolt_ampere"),
            rated_voltage=Voltage(7.2, "kilovolt"),
            voltage_type=VoltageTypes.LINE_TO_GROUND,
            connection_type=ConnectionType.STAR,
            resistance=0.6,
            is_grounded=True,
            tap_positions=[1.0, 1.0, 1.0],
        ),
        WindingEquipment(
            name=f"{prefix}_three_phase_secondary",
            num_phases=3,
            rated_power=ApparentPower(5000, "kilovolt_ampere"),
            rated_voltage=Voltage(120, "volt"),
            voltage_type=VoltageTypes.LINE_TO_GROUND,
            connection_type=ConnectionType.STAR,
            resistance=0.012,
            is_grounded=True,
            tap_positions=[1.0, 1.0, 1.0],
        ),
    ]
    catalog.add_components(*three_phase_windings)
    catalog.add_component(
        DistributionTransformerEquipment(
            name=f"{prefix}_three_phase_transformer",
            windings=three_phase_windings,
            is_center_tapped=False,
            pct_no_load_loss=0.1,
            pct_full_load_loss=1.0,
            coupling_sequences=[[0, 1]],
            winding_reactances=[0.02],
        )
    )

    return catalog


class ShiftExampleEdgeEquipmentMapper(EdgeEquipmentMapper):
    """Equipment mapper that supplies load and source equipment missing upstream."""

    @cached_property
    def edge_equipment_mapping(self):
        edge_equipment = {}
        for from_node, to_node, edge in self.graph.get_edges():
            served_load = self._get_served_load(from_node, to_node)
            from_phases = self.phase_mapper.node_phase_mapping[from_node] - {Phase.N}
            to_phases = self.phase_mapper.node_phase_mapping[to_node] - {Phase.N}
            num_phase = max(min(len(from_phases), len(to_phases)), 1)

            if issubclass(edge.edge_type, DistributionTransformer):
                edge_equipment[edge.name] = self._get_closest_transformer_equipment(
                    served_load,
                    num_phase,
                    [
                        self.voltage_mapper.node_voltage_mapping[from_node],
                        self.voltage_mapper.node_voltage_mapping[to_node],
                    ],
                )
            elif issubclass(edge.edge_type, DistributionBranchBase):
                kv = self.voltage_mapper.node_voltage_mapping[from_node].to("kilovolt").magnitude
                kva = served_load.to("kilova").magnitude
                is_split_phase = Phase.S1 in from_phases or Phase.S2 in from_phases
                current = (
                    kva / kv
                    if num_phase == 1
                    else kva / (2 * kv)
                    if is_split_phase
                    else kva / (math.sqrt(3) * kv)
                )
                edge_equipment[edge.name] = self._get_closest_branch_equipment(
                    MatrixImpedanceBranchEquipment,
                    Current(current, "ampere"),
                    num_phase,
                )
        return edge_equipment

    @cached_property
    def node_asset_equipment_mapping(self):
        node_equipment = {}
        for node in self.graph.get_nodes():
            if not node.assets:
                continue
            node_map = {}
            phases = self.phase_mapper.node_phase_mapping[node.name] - {Phase.N}
            num_phase = max(len(phases), 1)

            if DistributionLoad in node.assets:
                node_map[DistributionLoad] = LoadEquipment(
                    name=f"load_equipment_{node.name}",
                    phase_loads=[
                        PhaseLoadEquipment(
                            name=f"phase_load_equipment_{node.name}_{idx}",
                            real_power=ActivePower(5 / num_phase, "kilowatt"),
                            reactive_power=ReactivePower(1 / num_phase, "kilovar"),
                            z_real=0.0,
                            z_imag=0.0,
                            i_real=0.0,
                            i_imag=0.0,
                            p_real=1.0,
                            p_imag=1.0,
                        )
                        for idx in range(num_phase)
                    ],
                )

            if DistributionVoltageSource in node.assets:
                voltage = self.voltage_mapper.node_voltage_mapping[node.name]
                node_map[DistributionVoltageSource] = VoltageSourceEquipment(
                    name=f"voltage_source_equipment_{node.name}",
                    sources=[
                        PhaseVoltageSourceEquipment(
                            name=f"phase_voltage_source_equipment_{node.name}_{idx}",
                            r0=Resistance(0.001, "ohm"),
                            r1=Resistance(0.001, "ohm"),
                            x0=Reactance(0.001, "ohm"),
                            x1=Reactance(0.001, "ohm"),
                            voltage=voltage,
                            voltage_type=VoltageTypes.LINE_TO_GROUND,
                            angle=Angle(120 * idx, "degree"),
                        )
                        for idx in range(max(num_phase, 3))
                    ],
                )

            node_equipment[node.name] = node_map
        return node_equipment
