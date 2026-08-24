"""OpenML helpers and the dataset IDs reported in Table 2 of the paper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sklearn.datasets import fetch_openml


PAPER_DATASET_IDS = (
    4,
    1480,
    29,
    714,
    851,
    772,
    43097,
    40704,
    902,
    1104,
    890,
    346,
    464,
    172,
    782,
    1504,
    44701,
    41,
)


def fetch_paper_dataset(
    data_id: int = 714,
    *,
    data_home: str | Path | None = None,
    as_frame: bool = True,
) -> Any:
    """Download one of the paper's datasets through scikit-learn/OpenML."""

    if data_id not in PAPER_DATASET_IDS:
        raise ValueError(
            f"OpenML data_id={data_id} is not listed in Table 2 of the paper"
        )
    return fetch_openml(
        data_id=data_id,
        data_home=None if data_home is None else str(data_home),
        as_frame=as_frame,
    )

