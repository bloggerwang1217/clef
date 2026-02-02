"""
MuseSyn Processor for clef-piano-base
=====================================

Processes MuseSyn MusicXML files to Humdrum kern format for training data preparation.

Pipeline:
1. Parse MusicXML with music21
2. Sanitize score (fix cross-staff, hidden notes, etc.)
3. Extract repeat structure (barlines, DaCapo/Segno, volta brackets)
4. Expand repeats at music21 level (if present)
5. Build rich repeat_map (expanded → original measure mapping)
6. Export to Humdrum kern via converter21
7. Clean kern sequence (remove visual tokens)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple  # noqa: F401

import converter21
import music21 as m21

# Register converter21 for Humdrum output support
converter21.register()

from src.score.sanitize_piano_score import sanitize_score
from src.score.clean_kern import clean_kern_sequence, extract_visual_from_sequence, strip_non_kern_spines
from src.score.sanitize_kern import extract_kern_measures
from src.score.expand_repeat import (
    expand_musesyn_score,
    extract_repeat_structure,
    build_musesyn_repeat_map,
)

logger = logging.getLogger(__name__)


class MuseSynProcessor:
    """Process MuseSyn MusicXML files to cleaned kern format.

    Uses converter21 for MusicXML → Humdrum conversion (better success rate than verovio).
    """

    # Valid preset values (same as HumSynProcessor)
    PRESETS = {"clef-piano-base", "clef-piano-full", None}

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        visual_dir: Optional[Path] = None,
        repeat_map_dir: Optional[Path] = None,
        preset: Optional[str] = None,
    ):
        """Initialize MuseSyn processor.

        Args:
            input_dir: Path to MuseSyn directory (data/datasets/MuseSyn)
            output_dir: Path to output directory for processed kern files
            visual_dir: Path to output directory for visual info JSON files.
                        If None, visual info is not saved.
            repeat_map_dir: Path to output directory for repeat map JSON files.
                        If None, repeat maps are not saved.
            preset: Processing preset:
                - "clef-piano-base": Remove all non-kern spines
                - "clef-piano-full": Keep **dynam spines
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
        self.repeat_map_dir = Path(repeat_map_dir) if repeat_map_dir else None
        if self.repeat_map_dir:
            self.repeat_map_dir.mkdir(parents=True, exist_ok=True)
        self.preset = preset

        # clef-piano-base: remove all non-kern; clef-piano-full: keep **dynam
        self.keep_dynam = preset == "clef-piano-full"

    def process_one(
        self, xml_path: Path
    ) -> Optional[Tuple[str, List[List[Dict[str, Any]]], Dict]]:
        """Process a single MusicXML file to kern.

        Args:
            xml_path: Path to the MusicXML file

        Returns:
            Tuple of (cleaned kern content, visual info, repeat_map),
            or None if processing failed.
            Visual info is extracted from converter21 output BEFORE cleaning.
        """
        try:
            # 1. Parse MusicXML
            score = m21.converter.parse(str(xml_path))

            # 2. Sanitize score (fix cross-staff issues, hidden notes, etc.)
            sanitize_score(score)

            # 3. Extract repeat structure BEFORE expansion (barlines,
            #    DaCapo/Segno/Fine, volta brackets).
            repeat_structure = extract_repeat_structure(score)

            # 4. Expand repeats at the music21 level if present.
            # converter21 output has repeat barlines (:|! / !|:) but no
            # Humdrum expansion labels (*>[A,A,B,...]), so kern-level
            # expansion via expand_kern_repeats cannot work.
            # Instead, expand in music21 before converting to kern.
            original_measures = len(
                repeat_structure["orig_measure_numbers"]
            )
            score_for_kern, has_repeats = expand_musesyn_score(score)

            # 5. Export to Humdrum kern via converter21
            kern_raw = score_for_kern.write("humdrum")

            # Read the written file content
            with open(kern_raw, "r", encoding="utf-8") as f:
                kern_content = f.read()

            # Clean up temp file
            Path(kern_raw).unlink(missing_ok=True)

            # 6. Strip non-kern spines (keep_dynam=True for clef-piano-full)
            kern_content = strip_non_kern_spines(kern_content, keep_dynam=self.keep_dynam)

            # 7. Extract visual info BEFORE cleaning (preserves stem/beam/position)
            visual_info = extract_visual_from_sequence(kern_content)

            # 8. Clean kern sequence (remove visual tokens)
            kern_cleaned = clean_kern_sequence(kern_content, warn_tuplet_ratio=False)

            # 9. Build repeat_map with rich mapping
            measures_gt = extract_kern_measures(kern_content=kern_cleaned)

            repeat_map = build_musesyn_repeat_map(
                repeat_structure=repeat_structure,
                expanded_score=score_for_kern,
                measures_gt=measures_gt,
                has_repeats=has_repeats,
                original_measure_count=original_measures,
            )

            return kern_cleaned, visual_info, repeat_map

        except Exception as e:
            logger.error(f"Error processing {xml_path}: {e}")
            return None

    def process_all(self) -> Dict[str, str]:
        """Process all MusicXML files in MuseSyn directory.

        Returns:
            Dictionary mapping {output_filename: status}
            Status is one of: "success", "error: <message>"
        """
        results: Dict[str, str] = {}

        # Find all MusicXML files
        xml_patterns = ["*.xml", "*.musicxml", "*.mxl"]
        xml_files = []
        for pattern in xml_patterns:
            xml_files.extend(self.input_dir.glob(f"**/{pattern}"))

        xml_files = sorted(set(xml_files))
        logger.info(f"Found {len(xml_files)} MusicXML files in MuseSyn")

        for xml_path in xml_files:
            # Create output filename
            output_name = f"musesyn_{xml_path.stem}.krn"

            try:
                result = self.process_one(xml_path)

                if result is None:
                    results[output_name] = "error: processing failed"
                    continue

                kern_cleaned, visual_info, repeat_map = result

                # Write cleaned kern (repeat-expanded)
                output_path = self.output_dir / output_name
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(kern_cleaned)

                # Write visual info if visual_dir is configured
                if self.visual_dir:
                    visual_path = self.visual_dir / output_name.replace(".krn", ".json")
                    with open(visual_path, "w", encoding="utf-8") as f:
                        json.dump(visual_info, f)

                # Write repeat map if repeat_map_dir is configured
                if self.repeat_map_dir:
                    map_path = self.repeat_map_dir / output_name.replace(".krn", ".json")
                    with open(map_path, "w", encoding="utf-8") as f:
                        json.dump(repeat_map, f, indent=2, ensure_ascii=False)

                results[output_name] = "success"

            except Exception as e:
                logger.error(f"Error processing {xml_path}: {e}")
                results[output_name] = f"error: {e}"

        # Summary
        success = sum(1 for v in results.values() if v == "success")
        errors = sum(1 for v in results.values() if v.startswith("error"))
        logger.info(f"MuseSyn processing complete: {success} success, {errors} errors")

        return results


def main():
    """CLI entry point for MuseSyn processing."""
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Process MuseSyn MusicXML files to kern")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/datasets/MuseSyn"),
        help="Input MuseSyn directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments/clef_piano_base/kern"),
        help="Output directory for processed kern files",
    )

    args = parser.parse_args()

    processor = MuseSynProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )

    results = processor.process_all()

    # Print summary
    print(f"\nProcessed {len(results)} files:")
    success = sum(1 for v in results.values() if v == "success")
    errors = sum(1 for v in results.values() if v.startswith("error"))
    print(f"  success: {success}")
    print(f"  errors: {errors}")

    # Print error details
    if errors > 0:
        print("\nError details:")
        for name, status in results.items():
            if status.startswith("error"):
                print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
