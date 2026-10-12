from utils.scenario_loader import scenario_title


def test_scenario_title():
    assert scenario_title("augsburg_sb01") == "Augsburg SB 1"
    assert scenario_title("monteroni_di_lecce_sb12", "MPC", 3, 20) == "MPC: Monteroni Di Lecce SB 12   (3 of 20)"
