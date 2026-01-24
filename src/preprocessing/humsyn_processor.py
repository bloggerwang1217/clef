"""
HumSyn Processor for clef training data
=======================================

Processes HumSyn kern files for training data preparation.

Supported repositories:
- beethoven-piano-sonatas
- haydn-piano-sonatas
- mozart-piano-sonatas
- joplin
- scarlatti-keyboard-sonatas
- humdrum-chopin-first-editions

Processing modes (via preset parameter):
- "clef-piano-base": Chopin filter + Joplin remove **dynam & repeats
- "clef-piano-full": Chopin filter only
- None (default): No filtering (clef-tutti)
"""

import json
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.score.clean_kern import (
    clean_kern_sequence,
    extract_visual_from_sequence,
    strip_non_kern_spines,
)
from src.score.kern_zeng_compat import expand_tuplets_to_zeng_vocab

logger = logging.getLogger(__name__)


class HumSynProcessor:
    """Process HumSyn kern files for clef training.

    Handles:
    - Chopin filtering via selected_chopin.txt (clef-piano-base, clef-piano-full)
    - Joplin special processing (clef-piano-base only)
    - Unified kern cleaning via clean_kern_sequence()
    """

    # Files to exclude from processing (always excluded)
    EXCLUDED: Set[str] = {"school.krn"}

    # HumSyn repository names
    REPOS = [
        "beethoven-piano-sonatas",
        "haydn-piano-sonatas",
        "mozart-piano-sonatas",
        "joplin",
        "scarlatti-keyboard-sonatas",
        "humdrum-chopin-first-editions",
    ]

    # Valid preset values
    PRESETS = {"clef-piano-base", "clef-piano-full", None}

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        visual_dir: Optional[Path] = None,
        selected_chopin_path: Optional[Path] = None,
        preset: Optional[str] = None,
    ):
        """Initialize HumSyn processor.

        Args:
            input_dir: Path to HumSyn directory (data/datasets/HumSyn)
            output_dir: Path to output directory for processed kern files
            visual_dir: Path to output directory for visual info JSON files.
                        If None, visual info is not saved.
            selected_chopin_path: Path to selected_chopin.txt for filtering
            preset: Processing preset:
                - "clef-piano-base": Chopin filter + Joplin remove **dynam & repeats
                - "clef-piano-full": Chopin filter only
                - None: No filtering (clef-tutti, default)
        """
        if preset not in self.PRESETS:
            raise ValueError(f"Invalid preset: {preset}. Must be one of {self.PRESETS}")

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visual_dir = Path(visual_dir) if visual_dir else None
        if self.visual_dir:
            self.visual_dir.mkdir(parents=True, exist_ok=True)
        self.preset = preset

        # Determine what filters to apply based on preset
        # Always filter Chopin (selected_chopin.txt) - no reason to include duplicates
        self.filter_chopin = True
        self.strip_joplin = preset == "clef-piano-base"
        # clef-piano-base: expand septuplets to tied binary (Zeng vocab workaround)
        self.expand_septuplets = preset == "clef-piano-base"
        # clef-piano-base: remove all non-kern; clef-piano-full: keep **dynam
        self.keep_dynam = preset == "clef-piano-full"

        # Load Chopin selection list (when filtering is enabled)
        self.selected_chopin: Optional[Set[str]] = None
        if self.filter_chopin and selected_chopin_path and Path(selected_chopin_path).exists():
            self.selected_chopin = self._load_selected_chopin(selected_chopin_path)
            logger.info(f"Loaded {len(self.selected_chopin)} selected Chopin pieces")

    def _load_selected_chopin(self, path: Path) -> Set[str]:
        """Load selected Chopin filenames from text file."""
        selected = set()
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Remove .krn extension if present
                    if line.endswith(".krn"):
                        line = line[:-4]
                    selected.add(line)
        return selected

    def _strip_joplin_extras(self, kern_raw: str) -> str:
        """Strip Joplin-specific extras for clef-piano-base.

        Applied only when preset="clef-piano-base":
        - Remove **dynam spine (column 3)
        - Remove repeat markers (*>[...]) - skip entire line
        - Remove section markers (*>X) - skip entire line

        Args:
            kern_raw: Raw kern file content

        Returns:
            Processed kern content with only **kern spines
        """
        lines = kern_raw.split("\n")
        processed_lines = []

        for line in lines:
            # Skip empty lines
            if not line.strip():
                processed_lines.append(line)
                continue

            # Skip repeat markers: *>[I,A,A1,...] or *>norep[...]
            if re.match(r"^\*>\[.*\]", line) or re.match(r"^\*>norep\[.*\]", line):
                continue

            # Check if line has tabs (spine data)
            if "\t" in line:
                parts = line.split("\t")

                # Skip section markers (*>A, *>I, etc.) - all columns have same marker
                if parts[0].startswith("*>") and not parts[0].startswith("*>["):
                    continue

                # Remove third column (**dynam) if it exists
                if len(parts) >= 3:
                    # Keep only first two columns (left hand, right hand)
                    parts = parts[:2]

                processed_lines.append("\t".join(parts))
            else:
                # Non-tabbed lines (comments, etc.)
                processed_lines.append(line)

        return "\n".join(processed_lines)

    def _should_process_chopin(self, filename: str) -> bool:
        """Check if Chopin file should be processed based on selection."""
        if not self.filter_chopin or self.selected_chopin is None:
            return True

        # Remove .krn extension
        name = filename[:-4] if filename.endswith(".krn") else filename
        return name in self.selected_chopin

    def process_one(
        self, krn_path: Path, repo_name: str
    ) -> Optional[Tuple[str, List[List[Dict[str, Any]]]]]:
        """Process a single kern file.

        Args:
            krn_path: Path to the kern file
            repo_name: Name of the HumSyn repository

        Returns:
            Tuple of (cleaned kern content, visual info), or None if file should be skipped.
            Visual info is extracted BEFORE cleaning to preserve stem/beam/position markers.
        """
        filename = krn_path.name

        # Check exclusion list (always applied)
        if filename in self.EXCLUDED:
            logger.info(f"Skipping excluded file: {filename}")
            return None

        # Check Chopin selection (clef-piano-base, clef-piano-full)
        if repo_name == "humdrum-chopin-first-editions":
            if not self._should_process_chopin(filename):
                return None

        # Read raw kern file
        with open(krn_path, "r", encoding="utf-8", errors="replace") as f:
            kern_raw = f.read()

        # Apply Joplin-specific stripping (clef-piano-base only)
        if repo_name == "joplin" and self.strip_joplin:
            kern_raw = self._strip_joplin_extras(kern_raw)

        # Strip non-kern spines (keep_dynam=True for clef-piano-full)
        kern_raw = strip_non_kern_spines(kern_raw, keep_dynam=self.keep_dynam)

        # CRITICAL: Extract visual info BEFORE cleaning (preserves stem/beam/position)
        visual_info = extract_visual_from_sequence(kern_raw)

        # Apply unified kern cleaning (removes visual markers)
        kern_cleaned = clean_kern_sequence(kern_raw, warn_tuplet_ratio=False)

        # Expand tuplets to Zeng vocab (clef-piano-base only)
        # This is a Zeng vocab workaround for non-standard tuplets
        if self.expand_septuplets:
            kern_cleaned = expand_tuplets_to_zeng_vocab(kern_cleaned)

        return kern_cleaned, visual_info

    def process_all(self) -> Dict[str, str]:
        """Process all HumSyn repositories.

        Returns:
            Dictionary mapping {output_filename: status}
            Status is one of: "success", "skipped", "error: <message>"
        """
        results: Dict[str, str] = {}

        for repo_name in self.REPOS:
            repo_path = self.input_dir / repo_name / "kern"

            if not repo_path.exists():
                logger.warning(f"Repository not found: {repo_path}")
                continue

            logger.info(f"Processing {repo_name}...")

            krn_files = sorted(repo_path.glob("*.krn"))
            for krn_path in krn_files:
                # Create output filename with repo prefix
                output_name = f"{repo_name.replace('-', '_')}_{krn_path.stem}.krn"

                try:
                    result = self.process_one(krn_path, repo_name)

                    if result is None:
                        results[output_name] = "skipped"
                        continue

                    kern_cleaned, visual_info = result

                    # Write cleaned kern
                    output_path = self.output_dir / output_name
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(kern_cleaned)

                    # Write visual info if visual_dir is configured
                    if self.visual_dir:
                        visual_path = self.visual_dir / output_name.replace(".krn", ".json")
                        with open(visual_path, "w", encoding="utf-8") as f:
                            json.dump(visual_info, f)

                    results[output_name] = "success"

                except Exception as e:
                    logger.error(f"Error processing {krn_path}: {e}")
                    results[output_name] = f"error: {e}"

        # Summary
        success = sum(1 for v in results.values() if v == "success")
        skipped = sum(1 for v in results.values() if v == "skipped")
        errors = sum(1 for v in results.values() if v.startswith("error"))
        logger.info(f"HumSyn processing complete: {success} success, {skipped} skipped, {errors} errors")

        return results


def main():
    """CLI entry point for HumSyn processing."""
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Process HumSyn kern files")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/datasets/HumSyn"),
        help="Input HumSyn directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments/clef_piano_base/kern"),
        help="Output directory for processed kern files",
    )
    parser.add_argument(
        "--selected-chopin",
        type=Path,
        default=Path("data/metadata/selected_chopin.txt"),
        help="Path to selected Chopin file list",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["clef-piano-base", "clef-piano-full"],
        default=None,
        help="Processing preset (default: no filtering)",
    )

    args = parser.parse_args()

    processor = HumSynProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        selected_chopin_path=args.selected_chopin,
        preset=args.preset,
    )

    results = processor.process_all()

    # Print summary
    print(f"\nProcessed {len(results)} files:")
    for status in ["success", "skipped"]:
        count = sum(1 for v in results.values() if v == status)
        print(f"  {status}: {count}")
    error_count = sum(1 for v in results.values() if v.startswith("error"))
    if error_count:
        print(f"  errors: {error_count}")


if __name__ == "__main__":
    main()
