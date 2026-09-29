import pytest

from kaggle_tools.env import competition_dir, dataset_dir, first_existing


def test_competition_dir_prefers_competitions_layout(tmp_path):
    (tmp_path / "competitions" / "comp").mkdir(parents=True)
    (tmp_path / "comp").mkdir()
    assert competition_dir("comp", input_root=tmp_path) == tmp_path / "competitions" / "comp"


def test_competition_dir_falls_back_to_local(tmp_path):
    local = tmp_path / "data"
    local.mkdir()
    assert competition_dir("comp", local_dir=local, input_root=tmp_path / "none") == local


def test_dataset_dir_tries_all_layouts(tmp_path):
    (tmp_path / "ds").mkdir()
    assert dataset_dir("ds", owner="me", input_root=tmp_path) == tmp_path / "ds"
    (tmp_path / "datasets" / "me" / "ds").mkdir(parents=True)
    assert dataset_dir("ds", owner="me", input_root=tmp_path) == tmp_path / "datasets/me/ds"


def test_first_existing_lists_tried_paths(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope"):
        first_existing([tmp_path / "nope"])
