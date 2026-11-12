# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Named historical faults for the archive property tests; discovered by suffix."""

from pathlib import Path

from samudra.test_utils.catalogs import Catalog, Mutation

DATA = Path(__file__).resolve().parents[1]

MUTATIONS = (
    Mutation(
        "scalar_chunk_key",
        'refs[f"{name}/0"] = refs.pop(f"{name}/")',
        "pass",
        "test_static_archive_preserves_scalars_and_zero_coordinates",
    ),
    Mutation(
        "storage_fill_masks_zeros",
        'metadata["fill_value"] = None',
        "pass",
        "test_static_archive_preserves_scalars_and_zero_coordinates",
    ),
    Mutation(
        "unescaped_registry",
        "store = LocalURIStore(store)",
        "pass",
        "test_local_source_names_do_not_change_payload",
    ),
    Mutation(
        "double_decode_local_key",
        "return self.store.head(unquote(path))",
        "return self.store.head(unquote(unquote(path)))",
        "test_local_source_names_do_not_change_payload",
    ),
    Mutation(
        "encoded_async_key",
        "return await self.store.get_range_async(unquote(path), **options)",
        "return await self.store.get_range_async(path, **options)",
        "test_local_source_names_do_not_change_payload",
    ),
    Mutation(
        "encoded_reference_path",
        "reference[0] = unquote(reference[0])",
        "pass",
        "test_local_source_names_do_not_change_payload",
    ),
    Mutation(
        "replace_competing_directory",
        "rename_directory_exclusive(directory, output)",
        "os.rename(directory, output)",
        "test_publication_schedule_never_clobbers_another_writer",
    ),
    Mutation(
        "ignore_selected_source_changes",
        "if source.snapshot() != snapshot:",
        "if False:",
        "test_source_mutations_abort_publication[modify]",
    ),
    Mutation(
        "missing_chunk_shape_diagnostic",
        "if current_chunks != chunks:",
        "if False:",
        "test_native_file_partitions_round_trip_or_report_layout_error",
    ),
    Mutation(
        "missing_partial_chunk_diagnostic",
        "index < len(datasets) - 1\n                and dataset.sizes[concat_dim] % current_chunks[axis]",
        "False",
        "test_nonfinal_partial_chunks_have_actionable_errors",
    ),
    Mutation(
        "accept_sparse_native_chunks",
        "if sum(1 for _ in array.manifest.iter_refs()) < grid_size:",
        "if False:",
        "test_sparse_native_chunks_are_rejected",
    ),
)

CATALOG = Catalog(
    project=DATA,
    module="ocean_preprocessing.virtualize_om4",
    target=DATA / "ocean_preprocessing" / "virtualize_om4.py",
    properties=DATA / "tests" / "test_virtual_archive_properties.py",
    mutations=MUTATIONS,
)
