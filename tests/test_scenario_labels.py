from utils.scenario_loader import load_scenario_any


def test_real_scenarios_are_labelled_by_place_and_index():
    road = load_scenario_any("augsburg_sb01")[0]
    assert road._bump_labels[0] == "Augsburg SB 1"
