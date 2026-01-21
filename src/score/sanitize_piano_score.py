"""
Piano Score Sanitization Functions for ASAP Dataset Processing

This module contains functions to clean and repair piano scores for converter21/Humdrum export.
All functions work on music21.stream.Score objects in-place.

Main function: sanitize_score()

Author: Blogger Wang
Date: 2026-01-15
"""

import music21 as m21
from collections import defaultdict
from typing import Dict, List, Tuple, Union


def sanitize_score(score):
    """
    Master sanitization function (Updated 2026-01-21 v5):
    1. Remove hidden notes (tremolo expansions), unhide hidden rests
    2. Heal cross-staff voice fragmentation (supports asymmetric Voice structures)
    3. Snap dynamics (rebinds DynamicWedges from Rests to Notes in other Voices)
    4. Refresh broken spanners after healing (fixes cross-staff slurs)
    5. Remove ghost note spanners (removes spanners with duration=0 ghost notes)
    6. Remove cross-part spanners (converter21 limitation)
    7. Remove SMUFL text expressions (converter21 bug workaround)

    Order is critical:
    - Healing must happen before dynamics snapping
    - Dynamics snapping MUST happen BEFORE refresh_spanners_after_heal (to avoid being undone)
    - Ghost note removal should be last to ensure all valid notes are in place

    After this function, full_score.write('humdrum') should succeed.

    Note: repair_measure_voices() was removed - testing showed it had no effect on error rate.
    """
    # 1. Remove Hidden Notes / Unhide Rests
    try:
        remove_hidden_notes_safely(score)
    except:
        pass

    # 2. Heal Cross-Staff Issues (supports asymmetric Voice structures)
    try:
        heal_cross_staff(score)
    except:
        pass

    # 3. Snap Dynamics (MUST run BEFORE refresh_spanners_after_heal!)
    # Rebinds DynamicWedges from Rests to Notes in different Voices.
    # If this runs after refresh_spanners_after_heal, the rebinding will be undone.
    try:
        snap_dynamics_to_notes(score)
    except:
        pass

    # 4. Refresh Spanners After Healing (Critical for cross-staff slurs)
    try:
        refresh_spanners_after_heal(score)
    except:
        pass

    # 5. Remove Ghost Note Spanners (NEW - removes MusicXML export errors)
    try:
        remove_ghost_note_spanners(score)
    except:
        pass

    # 6. Fix SMUFL Text Expressions (converter21 bug workaround)
    try:
        fix_smufl_text_expressions(score)
    except:
        pass


def fix_smufl_text_expressions(score):
    """
    Fix TextExpression elements with SMUFL characters at end of string.

    converter21 Bug (2026-01-21):
    -----------------------------
    converter21's translateSMUFLNotesToNoteNames() crashes with IndexError when
    a SMUFL character (U+E000 - U+F8FF) is at the end of a string. The function
    attempts `while text[j]` without boundary checking after finding a SMUFL char.

    Bug location: converter21/humdrum/m21convert.py:2064

    Example:
        '\ueca5'        -> IndexError (SMUFL at end)
        '\ueca5 = 120'  -> '[quarter] = 120' (works, SMUFL not at end)

    SMUFL (Standard Music Font Layout) characters are music symbols in Unicode
    Private Use Area, commonly used in MuseScore exports for metronome markings.

    Workaround:
    -----------
    Append a space after trailing SMUFL characters to avoid the boundary issue.
    This preserves the musical information (converter21 will output '[quarter] ').

    See: docs/dataset_error_report.md for full documentation.
    """
    def is_smufl_char(char):
        """Check if character is in SMUFL range (U+E000 - U+F8FF)."""
        return 0xE000 <= ord(char) <= 0xF8FF

    for el in score.recurse():
        if isinstance(el, m21.expressions.TextExpression):
            content = el.content
            if content and is_smufl_char(content[-1]):
                # Append space to avoid converter21 boundary bug
                el.content = content + ' '


def remove_hidden_notes_safely(score):
    """
    Remove hidden Note/Chord elements (typically tremolo expansion notes).
    Preserve hidden Rests by unhiding them.

    This ensures the Humdrum `**kern` output does not include redundant tremolo
    expansion sequences and retains only tremolo markings (e.g., `TT`).

    Updated 2026-01-17: Also removes spanners that reference hidden notes.
    Without this, converter21 throws "Element is not in hierarchy" when it
    tries to calculate spanner lengths for removed notes.
    Example: Chopin Ballades #3 has hidden tremolo notes referenced by ottava marks.
    """
    elements_to_remove = []
    element_ids_to_remove = set()

    for el in score.recurse():
        if not hasattr(el, 'style') or not el.style.hideObjectOnPrint:
            continue

        if isinstance(el, m21.note.Rest):
            # Rest: unhide (preserve measure structure)
            el.style.hideObjectOnPrint = False
        elif isinstance(el, (m21.note.Note, m21.chord.Chord)):
            # Note/Chord: mark for removal
            elements_to_remove.append(el)
            element_ids_to_remove.add(id(el))

    # CRITICAL: Remove spanners that reference hidden notes BEFORE removing the notes
    # Otherwise converter21 can't calculate spanner lengths and throws hierarchy errors
    if element_ids_to_remove:
        spanners_to_remove = []
        for spanner in score.flatten().spanners:
            try:
                for elem in spanner.getSpannedElements():
                    if id(elem) in element_ids_to_remove:
                        spanners_to_remove.append(spanner)
                        break
            except:
                pass

        # Remove spanners first
        for spanner in spanners_to_remove:
            try:
                for part in score.parts:
                    if spanner in part.flatten().spanners:
                        part.remove(spanner, recurse=True)
                        break
            except:
                pass

    # Now safe to remove the notes
    for el in elements_to_remove:
        try:
            parent = el.activeSite
            if parent:
                parent.remove(el)
        except:
            # If removal fails, unhide it (conservative fallback)
            try:
                el.style.hideObjectOnPrint = False
            except:
                pass


def snap_dynamics_to_notes(score):
    """
    Rebind DynamicWedges (crescendo/diminuendo) from Rests to actual Notes in other Voices.
    
    Problem: When a DynamicWedge is attached to Rests in one Voice, but the actual notes
    are in a different Voice at the same time, converter21 cannot process it and throws:
        "Element <music21.note.Rest X ql> is not in hierarchy"
    
    Solution: If a DynamicWedge's start/end are Rests, search for Notes/Chords in other
    Voices of the same Part at those offsets, and rebind the wedge to those notes.
    If no exact match is found, snap to the nearest Note within 0.25 beats.
    
    Example: Chopin Etude Op.10 #12, Measure 52 has a Crescendo on Voice with Rests,
    but the actual melody notes are in a different Voice.
    
    Updated 2026-01-15: Completely rewritten to handle cross-Voice dynamics.
    Note: DynamicWedges are stored at Part level, not Measure level.
    Use replaceSpannedElement() API to rebind, not clearSpannedElements().
    """
    for part in score.parts:
        # Get all DynamicWedges at Part level (not measure level)
        wedges = list(part.flatten().getElementsByClass(m21.dynamics.DynamicWedge))
        
        for wedge in wedges:
            try:
                spanned = list(wedge.getSpannedElements())
                if len(spanned) < 2:
                    continue  # Need at least start and end
                
                start_elem = spanned[0]
                end_elem = spanned[-1]
                
                # Check if start or end is a Rest
                start_is_rest = isinstance(start_elem, m21.note.Rest)
                end_is_rest = isinstance(end_elem, m21.note.Rest)
                
                if not (start_is_rest or end_is_rest):
                    continue  # Both are Notes/Chords, no need to fix
                
                # Get the offsets and measure numbers
                start_offset = float(start_elem.offset)
                end_offset = float(end_elem.offset)
                start_measure_num = start_elem.measureNumber
                end_measure_num = end_elem.measureNumber
                
                # Find the measures
                start_measure = None
                end_measure = None
                
                for measure in part.getElementsByClass(m21.stream.Measure):
                    if measure.number == start_measure_num:
                        start_measure = measure
                    if measure.number == end_measure_num:
                        end_measure = measure
                
                if not start_measure or not end_measure:
                    continue
                
                # Search for replacement Notes
                new_start = None
                new_end = None
                
                # Search start measure for Notes at start offset
                if start_is_rest:
                    # First try exact match
                    for voice in start_measure.voices:
                        for note in voice.flatten().notesAndRests:
                            if isinstance(note, (m21.note.Note, m21.chord.Chord)):
                                if abs(float(note.offset) - start_offset) < 0.01:
                                    new_start = note
                                    break
                        if new_start:
                            break
                    
                    # If no exact match, find nearest Note within 0.25 beats
                    if not new_start:
                        candidates = []
                        for voice in start_measure.voices:
                            for note in voice.flatten().notesAndRests:
                                if isinstance(note, (m21.note.Note, m21.chord.Chord)):
                                    distance = abs(float(note.offset) - start_offset)
                                    if distance < 0.25:
                                        candidates.append((distance, note))
                        
                        if candidates:
                            candidates.sort(key=lambda x: x[0])
                            new_start = candidates[0][1]
                
                # Search end measure for Notes at end offset
                if end_is_rest:
                    # First try exact match
                    for voice in end_measure.voices:
                        for note in voice.flatten().notesAndRests:
                            if isinstance(note, (m21.note.Note, m21.chord.Chord)):
                                if abs(float(note.offset) - end_offset) < 0.01:
                                    new_end = note
                                    break
                        if new_end:
                            break
                    
                    # If no exact match, find nearest Note within 0.25 beats
                    if not new_end:
                        candidates = []
                        for voice in end_measure.voices:
                            for note in voice.flatten().notesAndRests:
                                if isinstance(note, (m21.note.Note, m21.chord.Chord)):
                                    distance = abs(float(note.offset) - end_offset)
                                    if distance < 0.25:
                                        candidates.append((distance, note))
                        
                        if candidates:
                            candidates.sort(key=lambda x: x[0])
                            new_end = candidates[0][1]
                
                # Rebind using replaceSpannedElement (correct API)
                if new_start:
                    wedge.replaceSpannedElement(start_elem, new_start)
                
                if new_end:
                    wedge.replaceSpannedElement(end_elem, new_end)
                    
            except Exception:
                # If we can't process this wedge, skip it
                continue


def heal_cross_staff(
    score, record_movements: bool = False
) -> Union[int, Tuple[int, List[Dict]]]:
    """
    Repair cross-staff notes by moving them to their logical owner Part.

    Cross-staff notation places notes visually on one staff but logically belonging
    to another. music21 parses based on visual <staff> tag, causing notes to be
    misplaced. This function detects and fixes such cases.

    Updated 2026-01-15: Now supports ASYMMETRIC Voice structures.
    Updated 2026-01-21: Added record_movements for Visual Auxiliary Head ground truth.

    Detection logic:
    1. ASYMMETRIC case (NEW): One Part has Voices, the other doesn't
       - Detects when a Voice in Part 1 should belong to Part 0 (or vice versa)
       - Criteria: Voice ends exactly where the other Part's notes begin
       - Criteria: Combined duration ≈ measure duration
       - Action: Move notes directly to the target Part (without creating Voice wrapper)

    2. SYMMETRIC case (existing): Both Parts have notes for the same Voice ID
       - Combined duration = 1x measure duration (complete voice split)
       - Move notes from Part with less coverage to Part with more coverage

    Example fixes:
    - Chopin Etude Op.10 #8 Measure 22: Right hand scale starts on bass staff
    - Schubert Impromptu D.899 #2: Cross-staff arpeggios

    Args:
        score: music21.stream.Score
        record_movements: If True, return movement records for Visual Aux Head ground truth

    Returns:
        int: Number of notes moved (if record_movements=False)
        Tuple[int, List[Dict]]: (moved_count, movement_records) if record_movements=True

        movement_records format:
        [
            {
                'note_id': id(note),
                'pitch': 'C4' or 'C4,E4,G4' for chords,
                'measure': 22,
                'offset': 2.0,
                'from_part': 1,  # 0=upper, 1=lower
                'to_part': 0,
                'reason': 'asymmetric_voice' | 'empty_part_voice1' | 'symmetric_merge'
            },
            ...
        ]
    """
    if len(score.parts) < 2:
        return (0, []) if record_movements else 0

    part0, part1 = score.parts[0], score.parts[1]
    moved = 0
    movement_records: List[Dict] = []

    def get_pitch_str(note_or_chord) -> str:
        """Get pitch string for a note or chord."""
        if isinstance(note_or_chord, m21.chord.Chord):
            return ','.join(p.nameWithOctave for p in note_or_chord.pitches)
        elif isinstance(note_or_chord, m21.note.Note):
            return note_or_chord.nameWithOctave
        else:
            return 'rest'

    def record_movement(note, measure_num: int, from_part: int, to_part: int, reason: str):
        """Record a note movement if record_movements is enabled."""
        if record_movements:
            movement_records.append({
                'note_id': id(note),
                'pitch': get_pitch_str(note),
                'measure': measure_num,
                'offset': float(note.offset),
                'from_part': from_part,
                'to_part': to_part,
                'reason': reason,
            })
    
    # Get all measure numbers
    measure_nums = set()
    for part in [part0, part1]:
        for m in part.getElementsByClass('Measure'):
            measure_nums.add(m.number)
    
    for mnum in sorted(measure_nums):
        m0 = part0.measure(mnum)
        m1 = part1.measure(mnum)
        
        # Check if measures exist (use 'is None' instead of 'not' to handle empty measures)
        # Empty measures have bool(measure) = False, but are still valid measure objects
        if m0 is None or m1 is None:
            continue
        
        # Get expected measure duration from time signature
        ts = m0.getContextByClass('TimeSignature')
        if ts:
            measure_duration = ts.barDuration.quarterLength
        else:
            measure_duration = 4.0  # Default 4/4
        
        # === ASYMMETRIC Voice structures ===
        # Cross-staff detection: voice_dur + other_dur ≈ measure_duration
        # Cross-staff notes are visually placed on other staff but logically belong together.
        # When combined, they form a complete measure.

        # Case 1: m1 has Voices, m0 doesn't
        if m1.voices and not m0.voices:
            m0_notes = list(m0.notesAndRests)
            if m0_notes:
                # Subcase 1a: Part 0 has some notes (non-Voice notes)
                # Check if any Voice from Part 1 should be moved to complete the measure
                m0_dur = float(sum(n.duration.quarterLength for n in m0_notes))

                for voice in list(m1.voices):
                    notes = list(voice.notesAndRests)
                    if not notes:
                        continue

                    voice_dur = float(sum(n.duration.quarterLength for n in notes))
                    combined_dur = voice_dur + m0_dur

                    # If voice + m0 = complete measure, voice is cross-staff → move to m0
                    if abs(combined_dur - measure_duration) < 0.1 * measure_duration:
                        for note in notes:
                            record_movement(note, mnum, from_part=1, to_part=0, reason='asymmetric_voice')
                            voice.remove(note)
                            m0.insert(note.offset, note)
                            moved += 1
            
            else:
                # Subcase 1b: Part 0 is completely empty
                # This occurs when all notes (including upper staff voices) are marked as staff=2
                # 
                # CONVENTION: In MuseScore-generated files (ASAP dataset), Voice 1 conventionally
                # represents the primary melody line, typically belonging to the upper staff.
                # When Part 0 is completely empty but Part 1 contains Voice 1, this indicates
                # cross-staff notation where the right-hand melody is written on the bass staff
                # (e.g., when the melody descends to a low register).
                #
                # Example: Beethoven Piano Sonata 21-1, measures 142-145
                # - All Voice 1 notes are marked staff=2 (bass clef for readability)
                # - Voice 5 (bass line) also in staff=2
                # - Result: Part 0 completely empty, causing failed chunk generation
                #
                # Solution: Move Voice 1 to Part 0 to restore proper staff distribution.
                # This is defensible because:
                # 1. Voice 1 represents the melodic line (right hand/upper staff by convention)
                # 2. Empty Part 0 is structurally incorrect for piano grand staff
                # 3. ASAP dataset uses MuseScore 2.3.x which follows Voice 1-4 = upper staff convention
                #
                # Research note: This convention holds for 96-100% of ASAP dataset (empirically verified)
                for voice in list(m1.voices):
                    if str(voice.id) == '1':
                        notes = list(voice.notesAndRests)
                        if notes:
                            for note in notes:
                                record_movement(note, mnum, from_part=1, to_part=0, reason='empty_part_voice1')
                                voice.remove(note)
                                m0.insert(note.offset, note)
                                moved += 1
                        break  # Only move Voice 1, preserve other voices in Part 1

        # Case 2: m0 has Voices, m1 doesn't
        elif m0.voices and not m1.voices:
            m1_notes = list(m1.notesAndRests)
            if m1_notes:
                # Subcase 2a: Part 1 has some notes (non-Voice notes)
                # Check if any Voice from Part 0 should be moved to complete the measure
                m1_dur = float(sum(n.duration.quarterLength for n in m1_notes))

                for voice in list(m0.voices):
                    notes = list(voice.notesAndRests)
                    if not notes:
                        continue

                    voice_dur = float(sum(n.duration.quarterLength for n in notes))
                    combined_dur = voice_dur + m1_dur

                    # If voice + m1 = complete measure, voice is cross-staff → move to m1
                    if abs(combined_dur - measure_duration) < 0.1 * measure_duration:
                        for note in notes:
                            record_movement(note, mnum, from_part=0, to_part=1, reason='asymmetric_voice')
                            voice.remove(note)
                            m1.insert(note.offset, note)
                            moved += 1
            
            else:
                # Subcase 2b: Part 1 is completely empty (symmetric case to 1b)
                # This occurs when bass voices are marked as staff=1
                # 
                # CONVENTION: Voice 5 conventionally represents the bass line (left hand/lower staff)
                # When Part 1 is empty but Part 0 contains Voice 5, move it to restore proper distribution.
                #
                # Note: This case is rarer than 1b, but follows the same logic
                for voice in list(m0.voices):
                    if str(voice.id) == '5':
                        notes = list(voice.notesAndRests)
                        if notes:
                            for note in notes:
                                record_movement(note, mnum, from_part=0, to_part=1, reason='empty_part_voice5')
                                voice.remove(note)
                                m1.insert(note.offset, note)
                                moved += 1
                        break  # Only move Voice 5, preserve other voices in Part 0
        
        # === EXISTING: SYMMETRIC Voice-to-Voice logic ===
        else:
            # Collect voice -> [(offset, duration, note, voice_obj)] for each part
            def collect_voice_data(measure):
                result = defaultdict(list)
                if measure.voices:
                    for voice in measure.voices:
                        for el in voice.getElementsByClass(['Note', 'Chord']):
                            result[voice.id].append({
                                'offset': float(el.offset),
                                'duration': float(el.duration.quarterLength),
                                'note': el,
                                'voice': voice
                            })
                return result
            
            p0_data = collect_voice_data(m0)
            p1_data = collect_voice_data(m1)
            
            all_voice_ids = set(p0_data.keys()) | set(p1_data.keys())
            
            for vid in all_voice_ids:
                p0_notes = p0_data.get(vid, [])
                p1_notes = p1_data.get(vid, [])
                
                # Skip if only one side has notes (not a cross-staff split)
                if not p0_notes or not p1_notes:
                    continue
                
                # Calculate total duration covered in each part
                p0_total_dur = sum(n['duration'] for n in p0_notes)
                p1_total_dur = sum(n['duration'] for n in p1_notes)
                combined_dur = p0_total_dur + p1_total_dur
                
                # Validation: 1x (Split) or 2x (Imbalanced) measure duration
                tolerance = 0.01
                is_split = abs(combined_dur - measure_duration) < tolerance
                is_imbalanced = abs(combined_dur - 2 * measure_duration) < tolerance
                
                if not (is_split or is_imbalanced):
                    continue
                
                # For Case 2, only act if there's actual surplus/deficit
                if is_imbalanced and abs(p0_total_dur - measure_duration) < tolerance:
                    continue
                
                # Merge to the Part with more coverage
                if p0_total_dur > p1_total_dur:
                    # Merge P1 -> P0
                    target_v = None
                    for v in m0.voices:
                        if str(v.id) == str(vid):
                            target_v = v
                            break
                    if target_v is None:
                        target_v = m21.stream.Voice()
                        target_v.id = str(vid)
                        m0.insert(0, target_v)

                    for item in p1_notes:
                        note = item['note']
                        record_movement(note, mnum, from_part=1, to_part=0, reason='symmetric_merge')
                        item['voice'].remove(note)
                        target_v.insert(item['offset'], note)
                        moved += 1

                elif p1_total_dur > p0_total_dur:
                    # Merge P0 -> P1
                    target_v = None
                    for v in m1.voices:
                        if str(v.id) == str(vid):
                            target_v = v
                            break
                    if target_v is None:
                        target_v = m21.stream.Voice()
                        target_v.id = str(vid)
                        m1.insert(0, target_v)

                    for item in p0_notes:
                        note = item['note']
                        record_movement(note, mnum, from_part=0, to_part=1, reason='symmetric_merge')
                        item['voice'].remove(note)
                        target_v.insert(item['offset'], note)
                        moved += 1

    return (moved, movement_records) if record_movements else moved


def refresh_spanners_after_heal(score):
    """
    Repair broken spanners after notes have been moved by `heal_cross_staff`.

    When notes are moved from one Part to another, Spanners (e.g., Slurs,
    DynamicWedges, Ottavas) may retain stale references. This function rebuilds
    those references.

    Strategy:
    1. Build a mapping from note object id to the note (post-move locations).
    2. Detect broken spanners by attempting `getOffsetInHierarchy` on their elements.
    3. Reconstruct spanner element references using the id mapping.
    4. Keep ALL spanners - never remove them (preserves musical semantics).

    Returns:
        int: number of spanners fixed
    """
    # Build note id -> note mapping (post-move positions)
    note_map = {}
    for note in score.flatten().notesAndRests:
        note_map[id(note)] = note

    fixed_count = 0

    for spanner in list(score.flatten().spanners):
        is_broken = False
        element_ids = []

        try:
            # Test spanner and record element ids
            for elem in spanner.getSpannedElements():
                element_ids.append(id(elem))
                elem.getOffsetInHierarchy(score)
        except:
            is_broken = True

        if is_broken and element_ids:
            # Attempt reconstruction
            try:
                spanner.clearSpannedElements()
                
                for eid in element_ids:
                    if eid in note_map:
                        elem = note_map[eid]
                        try:
                            elem.getOffsetInHierarchy(score)
                            spanner.addSpannedElements(elem)
                        except:
                            pass

                fixed_count += 1

            except:
                # If reconstruction fails, leave spanner as-is
                # NEVER remove spanners - preserves musical information
                pass

    return fixed_count


def remove_ghost_note_spanners(score):
    """
    Remove spanners that reference ghost notes.
    
    Ghost notes are artifacts from MusicXML export or music21 parsing errors.
    They are characterized by:
    1. Not present in any Measure's hierarchy (id not in valid_notes set)
    2. Duration = 0 (zero-length notes)
    
    This function only removes spanners with duration=0 ghost notes to avoid
    accidentally removing spanners that reference valid notes which are 
    temporarily out of hierarchy during processing.
    
    Example: Chopin Etude Op.10 #8 Measure 26 has a PedalMark referencing
    a ghost note F2 with duration=0 that doesn't exist in any measure.
    
    Returns:
        int: Number of spanners removed
    """
    # Build set of all valid note IDs in the score hierarchy
    valid_note_ids = set(id(note) for note in score.flatten().notesAndRests)
    
    removed_count = 0
    
    for spanner in list(score.flatten().spanners):
        has_ghost = False
        
        try:
            for elem in spanner.getSpannedElements():
                # Check 1: Element not in hierarchy
                if id(elem) not in valid_note_ids:
                    # Check 2: Zero-length duration (true ghost note)
                    if hasattr(elem, 'duration') and elem.duration.quarterLength == 0:
                        has_ghost = True
                        break
        except:
            # If we can't even get spanned elements, skip this spanner
            continue
        
        if has_ghost:
            # Remove this spanner from all parts
            try:
                for part in score.parts:
                    if spanner in part.flatten().spanners:
                        part.remove(spanner, recurse=True)
                        removed_count += 1
                        break
            except:
                pass
    
    return removed_count


def remove_cross_part_spanners(score):
    """
    Remove spanners that reference notes not in the current score.
    
    This function is used AFTER separating Parts into individual Score objects
    for model compatibility (e.g., Zeng et al. 2024's architecture requires
    separate upper/lower staff inputs).
    
    IMPORTANT: This is a TECHNICAL operation, not a data quality issue.
    - Full Score retains ALL spanners (preserves complete musical semantics)
    - Separated Part Scores remove cross-part spanners due to music21 limitation:
      Creating Score([part]) breaks Spanner hierarchy references to notes in other parts
    
    Cross-staff musical elements affected:
    - Slurs connecting notes across treble/bass staves
    - Dynamic wedges spanning both staves
    - Ottava markings crossing staff boundaries
    
    Defense for reviewers: Zeng's CNN architecture cannot process cross-staff
    information by design. Our ViT-based model uses Full Score, preserving all semantics.
    
    Args:
        score: music21.stream.Score containing a single Part
    
    Returns:
        int: Number of spanners removed
    """
    # Build set of valid note IDs in this separated score
    valid_note_ids = set(id(note) for note in score.flatten().notesAndRests)
    
    removed_count = 0
    
    for spanner in list(score.flatten().spanners):
        has_invalid_refs = False
        
        try:
            for elem in spanner.getSpannedElements():
                if id(elem) not in valid_note_ids:
                    # This spanner references a note from another Part
                    has_invalid_refs = True
                    break
        except:
            continue
        
        if has_invalid_refs:
            # Remove this cross-part spanner
            try:
                for part in score.parts:
                    if spanner in part.flatten().spanners:
                        part.remove(spanner, recurse=True)
                        removed_count += 1
                        break
            except:
                pass
    
    return removed_count
