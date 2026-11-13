# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Faults from the archive work that the virtualization properties must detect.

`python -m samudra_verification.gate` finds this file by its `_faults.py` suffix.
"""

from pathlib import Path

from samudra_verification.faults import Catalog, Fault

DATA = Path(__file__).resolve().parents[1]
PROPERTIES = "tests/test_virtual_archive_properties.py"


def archive_fault(name, before, after, *detectors):
    """A fault in `virtualize_om4`, detected by the named archive properties."""
    return Fault(
        name,
        "ocean_preprocessing.virtualize_om4",
        before,
        after,
        tuple(f"{PROPERTIES}::{test}" for test in detectors),
    )


FAULTS = (
    archive_fault(
        "scalar_chunk_key",
        'refs[f"{name}/0"] = refs.pop(f"{name}/")',
        "pass",
        "test_static_archive_preserves_scalars_and_zero_coordinates",
    ),
    archive_fault(
        "storage_fill_masks_zeros",
        'metadata["fill_value"] = None',
        "pass",
        "test_static_archive_preserves_scalars_and_zero_coordinates",
    ),
    archive_fault(
        "unescaped_registry",
        "store = LocalURIStore(store)",
        "pass",
        "test_local_source_names_do_not_change_payload",
    ),
    archive_fault(
        "double_decode_local_key",
        "return self.store.head(unquote(path))",
        "return self.store.head(unquote(unquote(path)))",
        "test_local_source_names_do_not_change_payload",
    ),
    archive_fault(
        "encoded_async_key",
        "return await self.store.get_range_async(unquote(path), **options)",
        "return await self.store.get_range_async(path, **options)",
        "test_local_source_names_do_not_change_payload",
    ),
    archive_fault(
        "encoded_reference_path",
        "reference[0] = unquote(reference[0])",
        "pass",
        "test_local_source_names_do_not_change_payload",
    ),
    archive_fault(
        "replace_competing_directory",
        "rename_directory_exclusive(directory, output)",
        "os.rename(directory, output)",
        "test_publication_schedule_never_clobbers_another_writer",
    ),
    archive_fault(
        "ignore_selected_source_changes",
        "if source.snapshot() != snapshot:",
        "if False:",
        "test_source_mutations_abort_publication[modify]",
    ),
    archive_fault(
        "missing_chunk_shape_diagnostic",
        "if current_chunks != chunks:",
        "if False:",
        "test_native_file_partitions_round_trip_or_report_layout_error",
    ),
    archive_fault(
        "missing_partial_chunk_diagnostic",
        "index < len(datasets) - 1\n                and dataset.sizes[concat_dim] % current_chunks[axis]",
        "False",
        "test_nonfinal_partial_chunks_have_actionable_errors",
    ),
    archive_fault(
        "accept_sparse_native_chunks",
        "if sum(1 for _ in array.manifest.iter_refs()) < grid_size:",
        "if False:",
        "test_sparse_native_chunks_are_rejected",
    ),
)

CATALOG = Catalog(project=DATA, faults=FAULTS)
