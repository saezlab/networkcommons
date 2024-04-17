import pathlib as pl
import pickle
import pandas as pd
import pytest

@pytest.fixture(scope="session")
def ca1_network() -> pd.DataFrame:

    with open(pl.Path("tests") / "_data" / "interactions.pickle", "rb") as fin:
        data: pd.DataFrame = pickle.load(fin)

    return data
