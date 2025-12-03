from __future__ import annotations

from pathlib import Path

from src.datastream.data_stream import (
    ORDER_22,
    LandmarkRecord,
    csv_to_trc_strict,
    header_fix_strict,
    write_landmark_csv,
)


def test_csv_to_trc_writes_strict_headers(tmp_path):
    csv_path = tmp_path / "sample.csv"
    records = [
        LandmarkRecord(0.0, "LHip", 0.0, 1.0, 2.0, 0.9),
        LandmarkRecord(0.0, "RHip", 0.5, 1.5, 2.5, 0.9),
    ]
    write_landmark_csv(csv_path, records)
    trc_path = tmp_path / "sample.trc"

    frames, markers = csv_to_trc_strict(csv_path, trc_path, ORDER_22)

    assert frames == 1
    assert markers == len(ORDER_22)
    lines = trc_path.read_text(encoding="utf-8").splitlines()
    assert "NumMarkers\t22" in lines[1]
    data_row = lines[5].split("\t")
    assert len(data_row) == 2 + 3 * len(ORDER_22)

    hip_idx = ORDER_22.index("Hip")
    hip_slice = slice(2 + hip_idx * 3, 2 + hip_idx * 3 + 3)
    assert data_row[hip_slice] != ["", "", ""]  # derived from LHip/RHip

    shoulder_idx = ORDER_22.index("RShoulder")
    shoulder_slice = slice(2 + shoulder_idx * 3, 2 + shoulder_idx * 3 + 3)
    assert data_row[shoulder_slice] == ["", "", ""]  # never observed → empty triplet


def test_header_fix_matches_triplet_counts(tmp_path):
    trc_path = tmp_path / "bad.trc"
    trc_path.write_text(
        "\n".join(
            [
                "PathFileType\t4\t(X/Y/Z)\tbad.trc",
                "DataRate\t30.00\tCameraRate\t30.00\tNumFrames\t1\tNumMarkers\t5\tUnits\tm",
                "",
                "Frame#\tTime\tHip\tHip\tHip\tHead\tHead\tHead",
                "\t\tX\tY\tZ\tX\tY\tZ",
                "1\t0.000000\t0.0\t1.0\t2.0\t1.0\t2.0\t3.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fixed_path = header_fix_strict(trc_path)
    lines = fixed_path.read_text(encoding="utf-8").splitlines()

    assert "NumMarkers\t2" in lines[1]
    assert lines[3].count("\t") + 1 == 2 + 3 * 2
    assert lines[4].count("X") == 2
    assert lines[5:] == trc_path.read_text(encoding="utf-8").splitlines()[5:]
