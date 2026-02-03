#!/usr/bin/env python3
"""
Round-trip test for kern reconstruction and MIDI/MusicXML conversion.

Tests 3 conversion paths using test set kern_gt files:
  A) Original kern_gt -> music21.parse -> write('midi')        [baseline]
  B) Reconstructed kern -> music21.parse -> write('midi')      [direct MIDI]
  C) Reconstructed kern -> music21.parse -> write('musicxml')  [via MusicXML]

Usage:
    poetry run python tests/test_reconstruct_kern.py
    poetry run python tests/test_reconstruct_kern.py --limit 5   # quick test
"""

import argparse
import json
import logging
import sys
import tempfile
import traceback
from pathlib import Path

import converter21
import music21

# Register converter21 for kern parsing
converter21.register()

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.clef.piano.tokenizer import KernTokenizer
from src.score.reconstruct_kern import extract_kern_metadata, reconstruct_kern_from_tokens

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def try_parse_kern(kern_content: str, label: str) -> tuple[music21.stream.Score | None, str]:
    """Try to parse kern content. Returns (score, error_msg).

    If an offset/timing error occurs, returns immediately since MIDI
    conversion will certainly fail.
    """
    try:
        score = music21.converter.parse(kern_content, format='humdrum')
        return score, ''
    except Exception as e:
        err = str(e)
        short = err[:120]
        if 'offset' in err.lower() or 'negative' in err.lower():
            logger.warning(f'  [{label}] Offset error (skip): {short}')
        else:
            logger.debug(f'  [{label}] Parse failed: {short}')
        return None, short


def try_write(score: music21.stream.Score, path: str, fmt: str) -> tuple[bool, str]:
    """Try to write score to a format. Returns (success, error_msg)."""
    try:
        score.write(fmt, fp=path)
        size = Path(path).stat().st_size
        return True, f'OK ({size:,} bytes)'
    except Exception as e:
        return False, str(e)[:120]


def test_one_file(
    kern_gt_path: Path,
    tokenizer: KernTokenizer,
    tmp_dir: Path,
    stem: str,
) -> dict:
    """Run round-trip test on a single kern_gt file."""
    result = {
        'id': stem,
        'n_tokens': 0,
        'n_unk': 0,
        'path_a_parse': False, 'path_a_midi': False, 'path_a_error': '',
        'path_b_parse': False, 'path_b_midi': False, 'path_b_error': '',
        'path_c_parse': False, 'path_c_musicxml': False, 'path_c_error': '',
    }

    kern_gt = kern_gt_path.read_text(encoding='utf-8')

    # --- Tokenize round-trip ---
    tokens = tokenizer.tokenize(kern_gt)
    result['n_tokens'] = len(tokens)
    result['n_unk'] = tokens.count('<unk>')

    # Extract metadata from original kern_gt for injection
    metadata = extract_kern_metadata(kern_gt)

    # Reconstruct from tokens (skip <sos>/<eos>), with metadata
    inner_tokens = [t for t in tokens if t not in ['<sos>', '<eos>']]
    reconstructed = reconstruct_kern_from_tokens(inner_tokens, metadata=metadata)

    # --- Path A: Original kern_gt -> MIDI ---
    score_a, err_a = try_parse_kern(kern_gt, 'A')
    if score_a:
        result['path_a_parse'] = True
        ok, msg = try_write(score_a, str(tmp_dir / f'{stem}_a.mid'), 'midi')
        result['path_a_midi'] = ok
        if not ok:
            result['path_a_error'] = msg
    else:
        result['path_a_error'] = f'parse: {err_a}'

    # --- Path B & C: Reconstructed kern (single parse, two outputs) ---
    score_r, err_r = try_parse_kern(reconstructed, 'B/C')
    if not score_r:
        # Parse failed -> both B and C fail, skip
        result['path_b_error'] = f'parse: {err_r}'
        result['path_c_error'] = f'parse: {err_r}'
        return result

    result['path_b_parse'] = True
    result['path_c_parse'] = True

    # Path B: write MIDI
    ok, msg = try_write(score_r, str(tmp_dir / f'{stem}_b.mid'), 'midi')
    result['path_b_midi'] = ok
    if not ok:
        result['path_b_error'] = msg

    # Path C: write MusicXML
    ok, msg = try_write(score_r, str(tmp_dir / f'{stem}_c.musicxml'), 'musicxml')
    result['path_c_musicxml'] = ok
    if not ok:
        result['path_c_error'] = msg

    return result


def main():
    parser = argparse.ArgumentParser(description='Round-trip test for kern reconstruction')
    parser.add_argument('--manifest', type=str,
                        default='data/experiments/clef_piano_base/test_manifest.json')
    parser.add_argument('--data-dir', type=str,
                        default='data/experiments/clef_piano_base')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of files to test (0 = all)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)
    logger.info(f'Loaded {len(manifest)} entries from manifest')

    # Deduplicate kern_gt paths (multiple soundfont versions share same kern_gt)
    kern_gt_set = {}
    for item in manifest:
        kern_gt_path = data_dir / item['kern_gt_path']
        if kern_gt_path not in kern_gt_set:
            kern_gt_set[kern_gt_path] = item['kern_gt_path']

    kern_gt_list = sorted(kern_gt_set.keys())
    if args.limit > 0:
        kern_gt_list = kern_gt_list[:args.limit]

    logger.info(f'Testing {len(kern_gt_list)} unique kern_gt files')

    tokenizer = KernTokenizer()
    results = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for i, kern_gt_path in enumerate(kern_gt_list):
            stem = kern_gt_path.stem
            logger.info(f'[{i+1}/{len(kern_gt_list)}] {stem}')

            try:
                r = test_one_file(kern_gt_path, tokenizer, tmp_path, stem)
                results.append(r)

                status = []
                if r['path_a_midi']:
                    status.append('A:MIDI')
                elif r['path_a_parse']:
                    status.append('A:parse-only')
                else:
                    status.append('A:FAIL')

                if r['path_b_midi']:
                    status.append('B:MIDI')
                elif r['path_b_parse']:
                    status.append('B:parse-only')
                else:
                    status.append('B:FAIL')

                if r['path_c_musicxml']:
                    status.append('C:XML')
                elif r['path_c_parse']:
                    status.append('C:parse-only')
                else:
                    status.append('C:FAIL')

                logger.info(f'  {" | ".join(status)} | tokens={r["n_tokens"]} unk={r["n_unk"]}')

            except Exception as e:
                logger.error(f'  Unexpected error: {e}')
                traceback.print_exc()
                results.append({'id': stem, 'error': str(e)})

    # --- Summary ---
    total = len(results)
    valid = [r for r in results if 'error' not in r]

    print('\n' + '=' * 70)
    print(f'ROUND-TRIP TEST SUMMARY ({total} files)')
    print('=' * 70)

    total_unk = sum(r.get('n_unk', 0) for r in valid)
    print(f'\nTotal <unk> tokens: {total_unk}')

    a_parse = sum(1 for r in valid if r['path_a_parse'])
    a_midi = sum(1 for r in valid if r['path_a_midi'])
    print(f'\nPath A (original kern_gt -> MIDI):')
    print(f'  Parse: {a_parse}/{len(valid)}')
    print(f'  MIDI:  {a_midi}/{len(valid)}')

    b_parse = sum(1 for r in valid if r['path_b_parse'])
    b_midi = sum(1 for r in valid if r['path_b_midi'])
    print(f'\nPath B (reconstructed -> MIDI):')
    print(f'  Parse: {b_parse}/{len(valid)}')
    print(f'  MIDI:  {b_midi}/{len(valid)}')

    c_parse = sum(1 for r in valid if r['path_c_parse'])
    c_xml = sum(1 for r in valid if r['path_c_musicxml'])
    print(f'\nPath C (reconstructed -> MusicXML):')
    print(f'  Parse: {c_parse}/{len(valid)}')
    print(f'  XML:   {c_xml}/{len(valid)}')

    # Failure details
    for label, key_parse, key_out, key_err in [
        ('A', 'path_a_parse', 'path_a_midi', 'path_a_error'),
        ('B', 'path_b_parse', 'path_b_midi', 'path_b_error'),
        ('C', 'path_c_parse', 'path_c_musicxml', 'path_c_error'),
    ]:
        fails = [r for r in valid if not r[key_out]]
        if fails:
            print(f'\n--- Path {label} failures ({len(fails)}) ---')
            for r in fails:
                print(f'  {r["id"]}: {r[key_err][:120]}')

    # Delta: B/C fails that A succeeds
    b_only = [r for r in valid if r['path_a_midi'] and not r['path_b_midi']]
    c_only = [r for r in valid if r['path_a_midi'] and not r['path_c_musicxml']]

    if b_only:
        print(f'\n--- B fails but A succeeds ({len(b_only)}) ---')
        for r in b_only:
            print(f'  {r["id"]}: {r["path_b_error"][:120]}')

    if c_only:
        print(f'\n--- C fails but A succeeds ({len(c_only)}) ---')
        for r in c_only:
            print(f'  {r["id"]}: {r["path_c_error"][:120]}')

    print('\n' + '=' * 70)


if __name__ == '__main__':
    main()
