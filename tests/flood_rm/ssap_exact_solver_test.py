from power.ssap import RootedFeeder, SsapEdge, solve_ssap_per_feeder


def test_tree_dp_matches_brute_force_ssap_objective_and_budget():
    feeder = RootedFeeder(
        root="source",
        loads_kw={
            "source": 0.0,
            "a": 12.0,
            "b": 8.0,
            "c": 5.0,
            "d": 7.0,
            "e": 4.0,
        },
        edges=(
            SsapEdge("source", "a", 1.0),
            SsapEdge("a", "b", 2.0),
            SsapEdge("a", "c", 1.5),
            SsapEdge("source", "d", 2.5),
            SsapEdge("d", "e", 1.0),
        ),
    )

    for k_switches in range(4):
        tree_dp = solve_ssap_per_feeder(feeder, k_switches=k_switches)
        brute_force = solve_ssap_per_feeder(
            feeder, k_switches=k_switches, algorithm="brute_force"
        )

        assert tree_dp.objective_value == brute_force.objective_value
        assert len(tree_dp.switch_edges) == k_switches
