import json

import pytest
import yaml

from utils.dataset import assign_split, place_snapshot, split_eval_names


def _crossing(path, low_confidence=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"name": path.stem, "bumps": [{"peak_z_m": 0.05, "low_confidence": low_confidence}]}))


def _place(src, out):
    train, seen, held = out / "train", out / "seen", out / "held"
    skipped = place_snapshot(src, (train, seen), (held,), out / "perception" / "perception_model.yaml")
    names = {d: {p.stem for p in (out / d).glob("*.json")} for d in ("train", "seen", "held")}
    return skipped, names


def test_without_an_eval_list_the_folders_decide():
    assert assign_split("a_sb01", "train", None) == "train"
    assert assign_split("a_sb01", "eval", None) == "eval"


def test_the_eval_list_overrides_the_folders():
    eval_names = {"a_sb01", "b_sb02"}
    assert assign_split("a_sb01", "train", eval_names) == "eval"
    assert assign_split("c_sb03", "eval", eval_names) == "train"
    assert assign_split("c_sb03", None, eval_names) == "train"
    assert assign_split("a_sb01", None, eval_names) == "eval"


def test_the_split_file_lists_the_eval_names(tmp_path):
    f = tmp_path / "split.yaml"
    f.write_text(yaml.safe_dump({"held_out_locations": ["x"], "eval": ["a_sb01", "b_sb02"]}))
    assert split_eval_names(f) == {"a_sb01", "b_sb02"}
    f.write_text(yaml.safe_dump({"held_out_locations": ["x"], "paper_split": {"eval": ["c_sb03"]}}))
    assert split_eval_names(f) == {"c_sb03"}
    f.write_text(yaml.safe_dump({"eval": ["a_sb01"], "paper_split": {"eval": ["c_sb03"]}}))
    assert split_eval_names(f) == {"a_sb01"}
    f.write_text(yaml.safe_dump({"held_out_locations": ["x"]}))
    assert split_eval_names(f) is None
    assert split_eval_names(tmp_path / "missing.yaml") is None


def test_flat_layout_sends_the_listed_roads_to_eval_and_the_rest_to_training(tmp_path):
    src = tmp_path / "snap"
    stems = {"germany": ["augsburg_sb07", "berlin_sb01", "berlin_sb02"], "italy": ["nardo_sb01", "carmiano_sb01"]}
    for region, names in stems.items():
        for s in names:
            _crossing(src / region / "crossings" / f"{s}.json")
    eval_names = ["augsburg_sb07", "nardo_sb01"]
    (src / "split.yaml").write_text(yaml.safe_dump({"held_out_locations": ["augsburg", "nardo"], "eval": eval_names}))
    _, names = _place(src, tmp_path / "out")
    everything = {s for v in stems.values() for s in v}
    assert names["held"] == set(eval_names)
    assert names["train"] == everything - set(eval_names)
    assert names["seen"] == names["train"]


def test_folder_layout_still_splits_by_folder_and_by_the_old_eval_list(tmp_path):
    src = tmp_path / "snap"
    _crossing(src / "germany" / "train" / "berlin_sb01.json")
    _crossing(src / "germany" / "eval" / "augsburg_sb01.json")
    _crossing(src / "italy" / "eval" / "nardo_sb01.json")
    _, names = _place(src, tmp_path / "out")
    assert names["held"] == {"augsburg_sb01", "nardo_sb01"}
    assert names["train"] == names["seen"] == {"berlin_sb01"}

    (src / "split.yaml").write_text(yaml.safe_dump({"paper_split": {"eval": ["nardo_sb01"]}}))
    _, names = _place(src, tmp_path / "out2")
    assert names["held"] == {"nardo_sb01"}
    assert names["train"] == {"berlin_sb01", "augsburg_sb01"}


def test_flat_layout_without_an_eval_list_fails_before_copying(tmp_path):
    src = tmp_path / "snap"
    _crossing(src / "germany" / "crossings" / "berlin_sb01.json")
    (src / "split.yaml").write_text(yaml.safe_dump({"held_out_locations": ["augsburg"]}))
    with pytest.raises(SystemExit, match="eval list"):
        _place(src, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_low_confidence_crossings_are_skipped(tmp_path):
    src = tmp_path / "snap"
    _crossing(src / "germany" / "crossings" / "berlin_sb01.json")
    _crossing(src / "germany" / "crossings" / "berlin_sb02.json", low_confidence=True)
    _crossing(src / "italy" / "crossings" / "nardo_sb01.json", low_confidence=True)
    (src / "split.yaml").write_text(yaml.safe_dump({"eval": ["nardo_sb01"]}))
    skipped, names = _place(src, tmp_path / "out")
    assert skipped == 2
    assert names["train"] == {"berlin_sb01"}
    assert names["held"] == set()


def test_the_combined_perception_model_is_copied_without_overwriting(tmp_path):
    src = tmp_path / "snap"
    _crossing(src / "germany" / "train" / "berlin_sb01.json")
    (src / "perception_model_de_it.yaml").write_text("max_range_m: 47.0\n")
    (src / "germany" / "perception_model").mkdir(parents=True)
    (src / "germany" / "perception_model" / "perception_model_de.yaml").write_text("max_range_m: 37.0\n")
    out = tmp_path / "out"
    _place(src, out)
    model = out / "perception" / "perception_model.yaml"
    assert model.read_text() == (src / "perception_model_de_it.yaml").read_text()
    model.write_text("mine\n")
    _place(src, out)
    assert model.read_text() == "mine\n"


def test_a_country_model_is_used_when_there_is_no_combined_one(tmp_path):
    src = tmp_path / "snap"
    _crossing(src / "germany" / "train" / "berlin_sb01.json")
    (src / "italy" / "perception_model").mkdir(parents=True)
    (src / "italy" / "perception_model" / "perception_model_it.yaml").write_text("max_range_m: 46.0\n")
    out = tmp_path / "out"
    _place(src, out)
    assert (out / "perception" / "perception_model.yaml").read_text() == "max_range_m: 46.0\n"
