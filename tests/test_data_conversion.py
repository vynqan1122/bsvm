from pathlib import Path

import pandas as pd

from roch_bsvm.data_conversion import (
    convert_file_to_csv,
    infer_target_column,
    looks_like_arff,
)


def test_infer_target_prefers_last_class_like_column():
    frame = pd.DataFrame(
        {
            "Class": [1, 2, 3],
            "feature": [0.1, 0.2, 0.3],
            "class": ["A", "B", "A"],
        }
    )
    assert infer_target_column(frame) == "class"


def test_convert_arff_without_extension_to_standard_csv(tmp_path: Path):
    source = tmp_path / "dataset_"
    source.write_text(
        "\n".join(
            [
                "@RELATION tiny",
                "@ATTRIBUTE x NUMERIC",
                "@ATTRIBUTE binaryClass {P,N}",
                "@DATA",
                "1,P",
                "2,N",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "csv"

    assert looks_like_arff(source)
    row = convert_file_to_csv(source, output_dir)
    converted = pd.read_csv(row["csv_path"])

    assert row["dataset"] == "dataset"
    assert row["classes"] == 2
    assert list(converted.columns) == ["x", "target"]
    assert converted["target"].tolist() == ["P", "N"]
