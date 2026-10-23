from utils.dataset import assign_split, paper_eval_names


def test_without_a_paper_split_the_folders_decide():
    assert assign_split("a_sb01", "train", None) == "train"
    assert assign_split("a_sb01", "eval", None) == "eval"


def test_the_paper_split_overrides_the_folders():
    eval_names = {"a_sb01", "b_sb02"}
    assert assign_split("a_sb01", "train", eval_names) == "eval"
    assert assign_split("c_sb03", "eval", eval_names) == "train"
    assert assign_split("c_sb03", "train", eval_names) == "train"


def test_every_scenario_lands_in_exactly_one_split():
    eval_names = {"a", "c"}
    for stem in ("a", "b", "c", "d"):
        for folder in ("train", "eval"):
            assert assign_split(stem, folder, eval_names) in ("train", "eval")


def test_the_split_file_lists_the_eval_names(tmp_path):
    f = tmp_path / "split.yaml"
    f.write_text("held_out_locations:\n  - x\npaper_split:\n  eval:\n    - a_sb01\n    - b_sb02\n")
    assert paper_eval_names(f) == {"a_sb01", "b_sb02"}
    f.write_text("held_out_locations:\n  - x\n")
    assert paper_eval_names(f) is None
    assert paper_eval_names(tmp_path / "missing.yaml") is None
