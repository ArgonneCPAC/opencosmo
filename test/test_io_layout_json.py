from __future__ import annotations

from opencosmo.header import read_header
from opencosmo.io.discover import (
    GroupLayout,
    LinkLayout,
    LinkSlot,
    LinkSlotKind,
    MapLayout,
    decode_file_layout_blob,
    encode_file_layout_blob,
)


def test_encode_decode_file_layout_roundtrip(test_data) -> None:
    header_path = test_data.snapshot.primary.halo_properties
    header = read_header(header_path)

    from uuid import UUID

    group = GroupLayout(
        path="/",
        header_path="/header",
        header=header,
        column_names=("a", "b"),
        column_dtypes=("int64", "float64"),
        column_units=(None, None),
        column_descriptions=(None, None),
        row_count=2,
        has_index=False,
        linked_target_names=("halo",),
        uuid=UUID("00000000-0000-0000-0000-000000000000"),
        has_persistent_uuid=False,
        link_layout=LinkLayout(
            path="/data_linked",
            slots=(
                LinkSlot(
                    prefix="halo",
                    kind=LinkSlotKind.CHUNKED,
                    dataset_names=("halo_start", "halo_size"),
                    length=3,
                ),
            ),
        ),
    )

    maps = MapLayout(
        path="/map",
        reference=UUID("12345678-1234-5678-1234-567812345678"),
        primary_slots=(
            (
                "87654321-4321-8765-4321-876543218765",
                UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            ),
        ),
        primary_lengths=((UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"), 7),),
        aux_slots=(
            (
                "b__c",
                UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            ),
        ),
    )

    from pathlib import Path

    from opencosmo.io.discover import FileLayout

    fl = FileLayout(
        path=Path("/tmp/example.hdf5"), groups=(group,), error=None, maps=(maps,)
    )
    blob = encode_file_layout_blob(fl)
    fl2 = decode_file_layout_blob(blob)
    assert fl2 is not None
    assert fl2.path == fl.path
    assert fl2.error == fl.error
    assert fl2.groups[0].path == fl.groups[0].path
    assert fl2.groups[0].header.to_dict() == fl.groups[0].header.to_dict()

    assert fl2.groups[0].link_layout is not None
    assert fl2.groups[0].link_layout.slots[0].kind == LinkSlotKind.CHUNKED


def test_decode_checksum_rejects() -> None:
    from pathlib import Path

    from opencosmo.io.discover import FileLayout

    fl = FileLayout(path=Path("/tmp/x.hdf5"), groups=(), error="boom", maps=())
    blob = encode_file_layout_blob(fl)
    assert "sha256" in blob
    blob["error"] = "changed"
    decoded = decode_file_layout_blob(blob)
    assert decoded is None
