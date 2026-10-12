import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import generate_ckd_dataset, load_dataset


def test_generate_ckd_dataset_shape(tmp_path):
    out_path = tmp_path / "test_raw.csv"
    df = generate_ckd_dataset(str(out_path), n=100, seed=1)
    assert df.shape[0] == 100 + 5   # 100 rows + 5 injected duplicates
    assert df.shape[1] == 16        # 15 features + target


def test_load_dataset_matches_saved_file(tmp_path):
    out_path = tmp_path / "test_raw.csv"
    generate_ckd_dataset(str(out_path), n=50, seed=1)
    loaded = load_dataset(str(out_path))
    assert len(loaded) == 55
    assert "class" in loaded.columns