import pytest

from cardiomoe.data.ptbxl import processed_dir
from cardiomoe.utils import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def _load(cfg, fname, loader):
    """Prefer the full prepared data; fall back to the --debug subset; skip if neither exists."""
    for debug in (False, True):
        if (processed_dir(cfg, debug) / fname).exists():
            return loader(cfg, debug)
    pytest.skip(f"{fname} not prepared yet")


@pytest.fixture(scope="session")
def ptbxl_meta(cfg):
    from cardiomoe.data import ptbxl
    return _load(cfg, "ptbxl_meta.parquet", ptbxl.load_meta)
