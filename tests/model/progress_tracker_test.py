import pytest
from _pytest.fixtures import fixture

from src.model.progress_tracker import ProgressTracker


@fixture
def directory(tmp_path):
    return tmp_path / "does" / "not" / "exist" / "yet"


@fixture
def tracker(directory):
    return ProgressTracker(directory)


def test_progress_tracker_creates_dir(directory, tracker):
    assert tracker.data_file_path.exists()
    assert directory == tracker.data_file_path.parent

    assert tracker.meta_data_file_path.exists()
    assert directory == tracker.meta_data_file_path.parent


def test_progress_tracker_raises_exception_if_files_exist_already(tmp_path):
    ProgressTracker(tmp_path)
    with pytest.raises(FileExistsError):
        ProgressTracker(tmp_path)


def test_progress_tracker_writes_metadata(tracker):
    tracker.write_metadata({"key": "value"})

    with open(tracker.meta_data_file_path, "r") as f:
        assert f.read() == '{"key": "value"}'


def test_progress_tracker_writes_header(tracker):
    with open(tracker.data_file_path, "r") as f:
        assert f.read() == "episode,number_of_moves,score,max_tile,epsilon\n\n"


def test_progress_tracker_appends_tracked_data(tracker):
    tracker.track(1, 1, 1, 1, 1)
    tracker.track(2, 2, 2, 2, 1)

    with open(tracker.data_file_path, "r") as f:
        f.readline()
        f.readline()
        assert f.readline() == "1,1,1,1,1\n"
        assert f.readline() == "2,2,2,2,1\n"
