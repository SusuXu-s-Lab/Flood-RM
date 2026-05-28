import pickle
from pathlib import Path

from gdm.distribution.components import DistributionBranchBase
from gdm.distribution.components import MatrixImpedanceBranch
from gdm.distribution.components import DistributionTransformer
from gdm.distribution.equipment import DistributionTransformerEquipment
from gdm.distribution.equipment import MatrixImpedanceBranchEquipment
from gdm.quantities import ApparentPower
from gdm.quantities import Voltage
from shift import BalancedPhaseMapper
from shift import DistributionGraph
from shift import DistributionSystemBuilder
from shift import TransformerPhaseMapperModel
from shift import TransformerTypes
from shift import TransformerVoltageMapper
from shift import TransformerVoltageModel

from power.shift_equipment import ShiftExampleEdgeEquipmentMapper
from power.shift_equipment import build_shift_example_equipment_catalog


repo_root = Path(__file__).resolve().parents[2]


def test_shift_example_equipment_catalog_supports_split_phase_and_three_phase_feeders():
    catalog = build_shift_example_equipment_catalog(prefix="test")

    branch_names = {
        branch.name
        for branch in catalog.get_components(MatrixImpedanceBranchEquipment)
    }
    transformers = {
        transformer.name: transformer
        for transformer in catalog.get_components(DistributionTransformerEquipment)
    }

    assert branch_names == {"test_branch_1ph", "test_branch_2ph", "test_branch_3ph"}
    assert transformers["test_split_phase_transformer"].is_center_tapped is True
    assert len(transformers["test_split_phase_transformer"].coupling_sequences) == 3
    assert transformers["test_three_phase_transformer"].is_center_tapped is False
    assert ShiftExampleEdgeEquipmentMapper.__name__ == "ShiftExampleEdgeEquipmentMapper"


def test_shift_example_equipment_mapper_builds_cached_prsg_graph_with_base_branch_edges():
    cache_path = repo_root / "locations/marshfield/data/power_grid/shift_cache/tiled_distribution_graphs.pkl"
    payload = pickle.loads(cache_path.read_bytes())
    graphs = payload["graphs"] if isinstance(payload, dict) else payload
    _, graph = next(iter(graphs.items()))

    assert any(edge.edge_type is DistributionBranchBase for _, _, edge in graph.get_edges())

    normalized_graph = DistributionGraph()
    for node in graph.get_nodes():
        normalized_graph.add_node(node)
    for from_node, to_node, edge in graph.get_edges():
        if edge.edge_type is DistributionBranchBase:
            edge.edge_type = MatrixImpedanceBranch
        normalized_graph.add_edge(from_node, to_node, edge_data=edge)
    graph = normalized_graph

    assert any(edge.edge_type is MatrixImpedanceBranch for _, _, edge in graph.get_edges())
    assert not any(edge.edge_type is DistributionBranchBase for _, _, edge in graph.get_edges())

    transformer_models = [
        TransformerPhaseMapperModel(
            tr_name=edge.name,
            tr_type=TransformerTypes.SPLIT_PHASE,
            tr_capacity=ApparentPower(25, "kilovolt_ampere"),
            location=graph.get_node(from_node).location,
        )
        for from_node, _, edge in graph.get_edges()
        if edge.edge_type is DistributionTransformer
    ]
    phase_mapper = BalancedPhaseMapper(graph, mapper=transformer_models, method="greedy")
    voltage_mapper = TransformerVoltageMapper(
        graph,
        xfmr_voltage=[
            TransformerVoltageModel(name=edge.name, voltages=[Voltage(7.2, "kilovolt"), Voltage(120, "volt")])
            for _, _, edge in graph.get_edges()
            if edge.edge_type is DistributionTransformer
        ],
    )
    catalog = build_shift_example_equipment_catalog(prefix="cached_graph")
    equipment_mapper = ShiftExampleEdgeEquipmentMapper(graph, catalog, voltage_mapper, phase_mapper)

    assert set(equipment_mapper.edge_equipment_mapping) == {edge.name for _, _, edge in graph.get_edges()}
    system = DistributionSystemBuilder(
        "cached_graph_smoke",
        graph,
        phase_mapper,
        voltage_mapper,
        equipment_mapper,
    ).get_system()
    assert len(list(system.iter_all_components())) > 0
