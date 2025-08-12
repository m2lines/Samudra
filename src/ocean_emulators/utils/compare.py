import filecmp
import mimetypes
from pathlib import Path

import numpy as np
from PIL import Image


def compare_images(file1, file2, diff_dir=None):
    """
    Compare two PNG images based on their RGB content.

    Args:
        file1: Path to first image
        file2: Path to second image
        diff_dir: Optional directory to save difference images

    Returns:
        bool: True if images have identical RGB content, False otherwise
    """
    try:
        # Open images and convert to RGB (ignoring alpha channel if present)
        img1 = Image.open(file1).convert("RGB")
        img2 = Image.open(file2).convert("RGB")

        # Check if sizes match
        if img1.size != img2.size:
            return False

        # Convert to numpy arrays and compare
        arr1 = np.array(img1)
        arr2 = np.array(img2)

        if np.array_equal(arr1, arr2):
            return True

        # If different and diff_dir is provided, create difference image
        if diff_dir:
            # Calculate absolute difference
            diff = np.abs(arr1.astype(np.int16) - arr2.astype(np.int16))

            # Enhance visibility by scaling up small differences
            diff_enhanced = np.clip(diff * 10, 0, 255).astype(np.uint8)

            # Create a composite image showing original1, original2, and difference
            height, width = arr1.shape[:2]
            composite = np.zeros((height, width * 3 + 20, 3), dtype=np.uint8)
            composite[:, :width] = arr1
            composite[:, width + 10 : width * 2 + 10] = arr2
            composite[:, width * 2 + 20 :] = diff_enhanced

            # Save the composite image
            diff_path = diff_dir / f"diff_{Path(file1).name}"
            Image.fromarray(composite).save(diff_path)

        return False
    except Exception as e:
        print(f"Error comparing images {file1} and {file2}: {e}")
        return False


def compare_directories(dir1: Path, dir2: Path, diff_dir: Path | None = None):
    """
    Compare two directories and return files that don't match.

    Args:
        dir1: Path to first directory
        dir2: Path to second directory
        diff_dir: Optional directory to save difference images

    Returns:
        tuple: (list of non-matching files, number of matching files)
    """
    dir1 = Path(dir1)
    dir2 = Path(dir2)

    if not dir1.exists():
        raise ValueError(f"Directory {dir1} does not exist")
    if not dir2.exists():
        raise ValueError(f"Directory {dir2} does not exist")

    # Get all files in both directories
    files1 = {f.relative_to(dir1) for f in dir1.rglob("*") if f.is_file()}
    files2 = {f.relative_to(dir2) for f in dir2.rglob("*") if f.is_file()}

    # Files only in one directory
    only_in_dir1 = files1 - files2
    only_in_dir2 = files2 - files1

    # Common files
    common_files = files1 & files2

    # Separate different types of non-matching files
    content_different = []
    missing_in_dir1 = []
    missing_in_dir2 = []
    matching_count = 0

    # Compare common files
    print(f"Comparing {len(common_files)} common files...")
    for f in common_files:
        file1 = dir1 / f
        file2 = dir2 / f

        # Special handling for image files
        file_mime, _encoding = mimetypes.guess_type(f)
        if file_mime and file_mime.startswith("image/"):
            if compare_images(file1, file2, diff_dir):
                matching_count += 1
            else:
                content_different.append(f)
        else:
            # Regular file comparison for non-image files
            if filecmp.cmp(file1, file2, shallow=False):
                matching_count += 1
            else:
                content_different.append(f)

    # Files only in one directory
    for f in only_in_dir1:
        missing_in_dir2.append(f)
    for f in only_in_dir2:
        missing_in_dir1.append(f)

    # Print summary
    print(f"\n=== Directory Comparison Summary ===")
    print(f"Total files in {dir1.name}: {len(files1)}")
    print(f"Total files in {dir2.name}: {len(files2)}")
    print(f"Common files: {len(common_files)}")
    print(f"Matching files: {matching_count}")

    # Print present but different files first
    if content_different:
        print(f"\nFiles with different content ({len(content_different)}):")
        _print_file_list(content_different)

    # Then print missing files
    if missing_in_dir2:
        print(f"\nFiles only in {dir1.name} ({len(missing_in_dir2)}):")
        _print_file_list(missing_in_dir2)

    if missing_in_dir1:
        print(f"\nFiles only in {dir2.name} ({len(missing_in_dir1)}):")
        _print_file_list(missing_in_dir1)

    # Combine all non-matching for return value
    non_matching = content_different + missing_in_dir1 + missing_in_dir2

    return non_matching, matching_count


def _print_file_list(file_list: list[Path]):
    for f in file_list[:10]:
        print(f"  - {f}")
    if len(file_list) > 10:
        print(f"  ... and {len(file_list) - 10} more")


if __name__ == "__main__":
    import argparse
    import sys
    import tempfile

    parser = argparse.ArgumentParser()
    parser.add_argument("dir1", type=Path)
    parser.add_argument("dir2", type=Path)
    args = parser.parse_args()

    diff_dir = Path(tempfile.mkdtemp("image-diffs"))
    non_matching, _ = compare_directories(args.dir1, args.dir2, diff_dir)
    print(f"See image differences in {diff_dir}")

    if len(non_matching) > 0:
        sys.exit(1)
