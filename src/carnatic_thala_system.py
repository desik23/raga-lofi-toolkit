#!/usr/bin/env python3
"""
Carnatic Tala System
-------------------
A module for generating authentic Carnatic tala patterns and rhythmic structures.
"""

import random
import mido
from mido import Message, MidiFile, MidiTrack
import os
import json
import time

class Anga:
    """
    Represents an anga (component) of a tala.
    
    In Carnatic music, talas are composed of three types of angas:
    - Laghu: Variable length, starts with a beat (clap) followed by finger counts
    - Drutam: Fixed length of 2 beats, consists of a clap and a wave
    - Anudrutam: Fixed length of 1 beat, just a clap
    """
    
    def __init__(self, anga_type, jati=4):
        """
        Initialize an anga.
        
        Parameters:
        - anga_type: "laghu", "drutam", or "anudrutam"
        - jati: For laghu, the number of counts (usually 3, 4, 5, 7, or 9)
        """
        self.anga_type = anga_type
        self.jati = jati
        
        # Calculate the length based on anga type
        if anga_type == "laghu":
            self.length = jati
        elif anga_type == "drutam":
            self.length = 2
        elif anga_type == "anudrutam":
            self.length = 1
        else:
            raise ValueError(f"Unknown anga type: {anga_type}")
    
    def get_pattern(self):
        """
        Get the beat pattern for this anga.
        
        Returns:
        - List of symbols representing claps ('C'), finger counts ('F'), waves ('W')
        """
        if self.anga_type == "laghu":
            return ['C'] + ['F'] * (self.jati - 1)
        elif self.anga_type == "drutam":
            return ['C', 'W']
        elif self.anga_type == "anudrutam":
            return ['C']
    
    def __repr__(self):
        if self.anga_type == "laghu":
            return f"Laghu({self.jati})"
        else:
            return self.anga_type.capitalize()


class Tala:
    """Represents a complete Carnatic tala."""
    
    # Common Carnatic talas
    COMMON_TALAS = {
        "adi": [("laghu", 4), ("drutam", None), ("drutam", None)],
        "rupaka": [("drutam", None), ("laghu", 4)],
        "misra_chapu": [("laghu", 3), ("laghu", 4)],
        "khanda_chapu": [("laghu", 5)],
        "eka": [("laghu", 4)],
        "tisra_eka": [("laghu", 3)],
        "tisra_triputa": [("laghu", 3), ("drutam", None), ("drutam", None)],
        "khanda_eka": [("laghu", 5)],
        "khanda_triputa": [("laghu", 5), ("drutam", None), ("drutam", None)],
        "misra_jhampa": [("laghu", 7), ("anudrutam", None), ("drutam", None)],
        "sankeerna_jathi": [("laghu", 9), ("laghu", 9), ("laghu", 9), ("laghu", 9)]
    }
    
    def __init__(self, name=None, angas=None, jati=4):
        """
        Initialize a tala.
        
        Parameters:
        - name: Name of a predefined tala, or None for custom
        - angas: List of anga types for custom tala, or None to use predefined
        - jati: Default jati (counts per laghu) for this tala
        """
        self.name = name
        self.jati = jati
        
        # Set up angas based on name or custom definition
        if name and name.lower() in self.COMMON_TALAS:
            tala_def = self.COMMON_TALAS[name.lower()]
            self.angas = []
            
            for anga_type, custom_jati in tala_def:
                # Use specified jati or default
                effective_jati = custom_jati if custom_jati else jati
                self.angas.append(Anga(anga_type, effective_jati))
        elif angas:
            # Custom defined angas
            self.angas = []
            for anga_def in angas:
                if isinstance(anga_def, Anga):
                    self.angas.append(anga_def)
                elif isinstance(anga_def, tuple):
                    anga_type, custom_jati = anga_def
                    effective_jati = custom_jati if custom_jati else jati
                    self.angas.append(Anga(anga_type, effective_jati))
                elif isinstance(anga_def, str):
                    self.angas.append(Anga(anga_def, jati))
                else:
                    raise ValueError(f"Unknown anga definition: {anga_def}")
        else:
            # Default to Adi tala if no specification
            self.name = "adi"
            self.angas = [
                Anga("laghu", jati),
                Anga("drutam"),
                Anga("drutam")
            ]
    
    def get_pattern(self):
        """
        Get the full beat pattern for this tala.
        
        Returns:
        - List of symbols representing the full tala cycle
        """
        pattern = []
        for anga in self.angas:
            pattern.extend(anga.get_pattern())
        return pattern
    
    def get_length(self):
        """Get the total length (number of beats) in this tala."""
        return sum(anga.length for anga in self.angas)
    
    def __repr__(self):
        if self.name:
            return f"{self.name.capitalize()} Tala (length: {self.get_length()})"
        else:
            angas_str = ", ".join(str(anga) for anga in self.angas)
            return f"Custom Tala: {angas_str} (length: {self.get_length()})"


class KonnakolPattern:
    """
    Represents rhythmic syllables (konnakol) used in Carnatic percussion.
    These syllables map to specific drum strokes.
    """
    
    # Standard konnakol syllables and their relative strengths
    SYLLABLES = {
        "tha": 1.0,    # Strong stroke
        "dhi": 0.9,    # Strong stroke
        "thom": 0.95,  # Strong stroke
        "nam": 0.8,    # Medium-strong stroke
        "ta": 0.7,     # Medium stroke
        "ki": 0.6,     # Medium-light stroke
        "mi": 0.5,     # Light stroke
        "ta": 0.7,     # Medium stroke
        "ka": 0.65,    # Medium stroke
        "di": 0.85,    # Strong-medium stroke
        "gi": 0.55,    # Light-medium stroke
        "na": 0.75,    # Medium stroke
        "num": 0.8,    # Medium-strong stroke
        "cha": 0.7,    # Medium stroke
        "lang": 0.9,   # Strong stroke
        "-": 0.0       # Rest
    }
    
    # Common konnakol patterns for different subdivisions
    COMMON_PATTERNS = {
        # Patterns for 3 subdivisions (tisra)
        3: [
            ["tha", "ki", "ta"],
            ["tha", "ka", "di"],
            ["tha", "-", "mi"]
        ],
        
        # Patterns for 4 subdivisions (chatusra)
        4: [
            ["tha", "ka", "dhi", "mi"],
            ["tha", "ki", "ta", "ka"],
            ["thom", "ka", "thom", "ka"],
            ["tha", "-", "dhi", "-"],
            ["dhi", "mi", "tha", "ka"]
        ],
        
        # Patterns for 5 subdivisions (khanda)
        5: [
            ["tha", "ka", "tha", "ki", "ta"],
            ["tha", "dhi", "mi", "tha", "ka"],
            ["tha", "ka", "dhi", "mi", "tha"],
            ["tha", "-", "ka", "-", "dhi"]
        ],
        
        # Patterns for 7 subdivisions (misra)
        7: [
            ["tha", "ka", "di", "mi", "tha", "ka", "di"],
            ["tha", "ka", "tha", "ki", "ta", "tha", "ka"],
            ["tha", "-", "ki", "-", "ta", "-", "ka"]
        ],
        
        # Patterns for 9 subdivisions (sankeerna)
        9: [
            ["tha", "ka", "di", "mi", "tha", "ka", "tha", "ki", "ta"],
            ["tha", "-", "ki", "-", "ta", "-", "ka", "-", "dhi"]
        ]
    }
    
    def __init__(self, subdivision=4, pattern=None):
        """
        Initialize a konnakol pattern.
        
        Parameters:
        - subdivision: Number of subdivisions per beat (3, 4, 5, 7, or 9)
        - pattern: Custom pattern of syllables, or None to use predefined
        """
        self.subdivision = subdivision
        
        # Set pattern
        if pattern:
            self.pattern = pattern
        else:
            # Use a predefined pattern if available
            if subdivision in self.COMMON_PATTERNS:
                self.pattern = random.choice(self.COMMON_PATTERNS[subdivision])
            else:
                raise ValueError(f"No predefined patterns for subdivision {subdivision}")
    
    def get_velocities(self, base_velocity=80):
        """
        Convert syllables to MIDI velocity values.
        
        Parameters:
        - base_velocity: Base MIDI velocity (0-127)
        
        Returns:
        - List of velocity values corresponding to syllables
        """
        return [
            int(base_velocity * self.SYLLABLES.get(syllable, 0.5))
            for syllable in self.pattern
        ]
    
    def __repr__(self):
        return " ".join(self.pattern)


class TalaPatternGenerator:
    """
    Generates rhythmic patterns based on Carnatic talas.
    """
    
    # MIDI note numbers for common percussion sounds
    MIDI_NOTES = {
        "mridangam_right": 60,  # Base drum sound
        "mridangam_left": 62,   # Bass drum sound
        "khanjira": 64,         # Frame drum (for variations)
        "kanjira_open": 65,     # Open frame drum
        "ghatam_low": 66,       # Low clay pot sound
        "ghatam_high": 67,      # High clay pot sound
        "tabla_na": 68,         # Tabla "Na" stroke
        "tabla_tin": 69,        # Tabla "Tin" stroke
        "tabla_dha": 70,        # Tabla "Dha" stroke
        "clap": 72,             # Hand clap sound
        "click": 74,            # Click/tick sound
        "rim": 76,              # Rim click
        "kit_kick": 36,         # Standard kit kick
        "kit_snare": 38,        # Standard kit snare
        "kit_hihat": 42,        # Standard kit closed hi-hat
        "kit_hihat_open": 46    # Standard kit open hi-hat
    }
    
    def __init__(self, use_western_kit=False):
        """
        Initialize the pattern generator.
        
        Parameters:
        - use_western_kit: If True, use standard drum kit sounds; otherwise use Indian percussion
        """
        self.use_western_kit = use_western_kit
    
    def generate_tala_pattern(self, tala, nadai=4, variations=True, length_multiplier=1):
        """
        Generate a complete tala pattern with appropriate nadai (subdivision).
        
        Parameters:
        - tala: Tala object defining the rhythmic cycle
        - nadai: Subdivision per beat (4 = chatusra, 3 = tisra, etc.)
        - variations: Whether to add rhythmic variations
        - length_multiplier: How many cycles to generate
        
        Returns:
        - Dictionary of percussion patterns for different instruments
        """
        tala_pattern = tala.get_pattern()
        tala_length = tala.get_length()
        
        # Determine which instruments to use
        instruments = self._get_instrument_mapping()
        
        # Initialize patterns for each instrument
        patterns = {name: [0] * (tala_length * nadai * length_multiplier) for name in instruments}
        
        # Generate base pattern first
        for cycle in range(length_multiplier):
            offset = cycle * tala_length * nadai
            
            # Process each beat in the tala
            beat_position = 0
            for symbol in tala_pattern:
                # Calculate the position in the full pattern
                position = offset + (beat_position * nadai)
                
                # Generate patterns for this beat based on its type
                if symbol == 'C':  # Clap (strong emphasis)
                    self._add_clap_pattern(patterns, position, nadai, is_strong=True)
                elif symbol == 'W':  # Wave (medium emphasis)
                    self._add_clap_pattern(patterns, position, nadai, is_strong=False)
                elif symbol == 'F':  # Finger count (light emphasis)
                    self._add_finger_pattern(patterns, position, nadai)
                
                beat_position += 1
        
        # Add variations if requested
        if variations:
            self._add_variations(patterns, tala_length, nadai, length_multiplier)
        
        return patterns
    
    def _get_instrument_mapping(self):
        """Get the appropriate instrument mapping based on settings."""
        if self.use_western_kit:
            return {
                "kick": self.MIDI_NOTES["kit_kick"],
                "snare": self.MIDI_NOTES["kit_snare"],
                "hihat": self.MIDI_NOTES["kit_hihat"],
                "hihat_open": self.MIDI_NOTES["kit_hihat_open"],
                "click": self.MIDI_NOTES["click"]
            }
        else:
            return {
                "mridangam_right": self.MIDI_NOTES["mridangam_right"],
                "mridangam_left": self.MIDI_NOTES["mridangam_left"],
                "kanjira": self.MIDI_NOTES["khanjira"],
                "kanjira_open": self.MIDI_NOTES["kanjira_open"],
                "ghatam": self.MIDI_NOTES["ghatam_low"],
                "clap": self.MIDI_NOTES["clap"]
            }
    
    def _add_clap_pattern(self, patterns, position, nadai, is_strong=True):
        """Add a pattern for a clap or wave beat."""
        # Choose a konnakol pattern for this subdivision
        konnakol = KonnakolPattern(nadai)
        velocities = konnakol.get_velocities(90 if is_strong else 70)
        
        # Add main beat emphasis
        if self.use_western_kit:
            # Western kit - use kick for strong beats, snare for medium
            patterns["kick"][position] = 100 if is_strong else 0
            patterns["snare"][position] = 80 if not is_strong else 0
            
            # Add hi-hat pattern
            for i in range(nadai):
                sub_pos = position + i
                if sub_pos < len(patterns["hihat"]):
                    # Every other subdivision gets a hi-hat
                    if i % 2 == 0:
                        patterns["hihat"][sub_pos] = max(patterns["hihat"][sub_pos], 70)
        else:
            # Indian percussion - use mridangam for main emphasis
            patterns["mridangam_right"][position] = 100 if is_strong else 80
            
            # Add ghatam on strong beats
            if is_strong:
                patterns["ghatam"][position] = 90
            
            # Kanjira for subdivision emphasis
            for i in range(nadai):
                sub_pos = position + i
                if sub_pos < len(patterns["kanjira"]):
                    if velocities[i % len(velocities)] > 0:
                        patterns["kanjira"][sub_pos] = velocities[i % len(velocities)]
    
    def _add_finger_pattern(self, patterns, position, nadai):
        """Add a pattern for a finger count beat (lighter emphasis)."""
        # Choose a konnakol pattern
        konnakol = KonnakolPattern(nadai)
        velocities = konnakol.get_velocities(60)  # Lower base velocity
        
        if self.use_western_kit:
            # Western kit - use lighter sounds
            patterns["hihat"][position] = 70
            
            # Add some soft snare hits for variations
            for i in range(nadai):
                sub_pos = position + i
                if sub_pos < len(patterns["snare"]) and velocities[i % len(velocities)] > 70:
                    patterns["snare"][sub_pos] = 60
        else:
            # Indian percussion - lighter mridangam and kanjira
            patterns["mridangam_right"][position] = 70
            
            # Add kanjira for subdivision accents
            for i in range(nadai):
                sub_pos = position + i
                if sub_pos < len(patterns["kanjira"]) and velocities[i % len(velocities)] > 0:
                    patterns["kanjira"][sub_pos] = velocities[i % len(velocities)]
    
    def _add_variations(self, patterns, tala_length, nadai, cycles):
        """Add rhythmic variations to make the pattern more interesting."""
        total_length = tala_length * nadai * cycles
        
        if self.use_western_kit:
            # Add kick drum variations
            self._add_kick_variations(patterns, total_length, nadai)
            
            # Add hi-hat variations
            self._add_hihat_variations(patterns, total_length, nadai)
            
            # Add fill variations at cycle boundaries
            for cycle in range(1, cycles):
                fill_pos = cycle * tala_length * nadai - nadai  # Just before cycle end
                self._add_drum_fill(patterns, fill_pos, nadai)
        else:
            # Add mridangam variations
            self._add_mridangam_variations(patterns, total_length, nadai)
            
            # Add kanjira variations
            self._add_kanjira_variations(patterns, total_length, nadai)
            
            # Add ghatam variations for cycle endings
            for cycle in range(1, cycles):
                fill_pos = cycle * tala_length * nadai - nadai
                self._add_ghatam_fill(patterns, fill_pos, nadai)
    
    def _add_kick_variations(self, patterns, total_length, nadai):
        """Add variations to kick drum pattern."""
        # Add occasional syncopated kicks
        for i in range(0, total_length, nadai):
            # 20% chance of adding a syncopated kick
            if i > nadai and random.random() < 0.2:
                # Add kick at an off-beat position
                off_beat = i + random.choice([1, 3]) if nadai == 4 else i + 1
                if off_beat < total_length and patterns["kick"][off_beat] == 0:
                    patterns["kick"][off_beat] = 70 + random.randint(-10, 10)
    
    def _add_hihat_variations(self, patterns, total_length, nadai):
        """Add variations to hi-hat pattern."""
        # Add occasional open hi-hats
        for i in range(0, total_length, nadai):
            # 15% chance of adding an open hi-hat
            if random.random() < 0.15:
                # Usually on off-beats
                off_beat = i + random.choice(list(range(1, nadai)))
                if off_beat < total_length:
                    patterns["hihat"][off_beat] = 0  # Remove closed hi-hat
                    patterns["hihat_open"][off_beat] = 70 + random.randint(-10, 10)
    
    def _add_drum_fill(self, patterns, position, nadai):
        """Add a drum fill near the end of a cycle."""
        # Simple drum fill using snare
        fill_length = min(nadai * 2, total_length - position)
        
        for i in range(fill_length):
            pos = position + i
            if pos < total_length:
                # Clear other drum sounds during fill
                patterns["kick"][pos] = 0
                patterns["hihat"][pos] = 0
                
                # Add snare hits with increasing velocity
                if i % 2 == 0 or i >= fill_length - 2:
                    patterns["snare"][pos] = 70 + (i * 5)
    
    def _add_mridangam_variations(self, patterns, total_length, nadai):
        """Add variations to mridangam patterns."""
        # Add left hand (bass) mridangam accents
        for i in range(0, total_length, nadai):
            if patterns["mridangam_right"][i] > 0 and random.random() < 0.4:
                # Add left hand on some strong beats
                patterns["mridangam_left"][i] = patterns["mridangam_right"][i] - 10
            
            # Add occasional rhythmic flourishes
            if i > nadai and random.random() < 0.15:
                # Add a quick double or triple stroke
                stroke_count = random.choice([2, 3])
                for j in range(stroke_count):
                    pos = i + j
                    if pos < total_length and patterns["mridangam_right"][pos] == 0:
                        patterns["mridangam_right"][pos] = 60 + random.randint(-10, 10)
    
    def _add_kanjira_variations(self, patterns, total_length, nadai):
        """Add variations to kanjira patterns."""
        # Add open kanjira accents occasionally
        for i in range(0, total_length, nadai * 2):  # Every other beat
            if random.random() < 0.3:
                # Choose an off-beat position
                off_beat = i + random.choice(list(range(1, nadai)))
                if off_beat < total_length:
                    patterns["kanjira"][off_beat] = 0  # Remove regular kanjira
                    patterns["kanjira_open"][off_beat] = 80 + random.randint(-10, 10)
    
    def _add_ghatam_fill(self, patterns, position, nadai):
        """Add a ghatam (clay pot) fill near the end of a cycle."""
        fill_length = min(nadai * 2, total_length - position)
        
        for i in range(fill_length):
            pos = position + i
            if pos < total_length:
                # Add alternating low and high ghatam sounds
                if i % 2 == 0:
                    patterns["ghatam"][pos] = 70 + (i * 3)
                else:
                    # Use a different note for high ghatam
                    patterns["ghatam"][pos] = 0
                    patterns["ghatam_high"][pos] = 75 + (i * 3)
    
    def create_midi_from_patterns(self, patterns, filename, bpm=75):
        """
        Create a MIDI file from the generated percussion patterns.
        
        Parameters:
        - patterns: Dictionary of percussion patterns (note values by position)
        - filename: Output filename
        - bpm: Tempo in beats per minute
        
        Returns:
        - Filename of the generated MIDI file
        """
        # Create MIDI file
        midi = MidiFile()
        track = MidiTrack()
        midi.tracks.append(track)
        
        # Add tempo
        tempo = mido.bpm2tempo(bpm)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))
        
        # Add track name
        track.append(mido.MetaMessage('track_name', name="Carnatic Rhythm"))
        
        # Set percussion channel (channel 9, zero-indexed is 10)
        channel = 9
        
        # Find the pattern length
        pattern_length = max(len(pattern) for pattern in patterns.values())
        
        # Create a time-ordered list of note events
        events = []
        
        for instrument, pattern in patterns.items():
            note = self.MIDI_NOTES.get(instrument, patterns[instrument])
            
            for pos, velocity in enumerate(pattern):
                if velocity > 0:
                    # Note on at this position
                    events.append((pos, 'note_on', note, velocity))
                    # Note off shortly after
                    events.append((pos + 0.5, 'note_off', note, 0))
        
        # Sort events by time
        events.sort()
        
        # Convert events to MIDI messages
        previous_time = 0
        for pos, event_type, note, velocity in events:
            # Calculate delta time (in ticks)
            delta_time = int((pos - previous_time) * midi.ticks_per_beat / 4)
            previous_time = pos
            
            # Add the message
            if event_type == 'note_on':
                track.append(Message('note_on', note=note, velocity=velocity, 
                                    channel=channel, time=max(0, delta_time)))
            else:  # note_off
                track.append(Message('note_off', note=note, velocity=0, 
                                    channel=channel, time=max(0, delta_time)))
        
        # Save MIDI file
        timestamp = int(time.time())
        if not filename:
            filename = f"outputs/carnatic_rhythm_{timestamp}.mid"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        midi.save(filename)
        return filename

class TalaLoFiAdapter:
    """
    Adapts traditional Carnatic talas for lo-fi beat production.
    """
    
    def __init__(self, use_western_kit=True):
        """
        Initialize the adapter.
        
        Parameters:
        - use_western_kit: Whether to use Western drum kit instead of Indian percussion
        """
        self.tala_generator = TalaPatternGenerator(use_western_kit)
    
    def generate_lo_fi_rhythm(self, tala_name="adi", cycles=4, bpm=75, nadai=4, filename=None):
        """
        Generate a lo-fi friendly rhythm track based on a Carnatic tala.
        
        Parameters:
        - tala_name: Name of the tala to use
        - cycles: Number of tala cycles to generate
        - bpm: Tempo in beats per minute
        - nadai: Subdivision level (3=tisra, 4=chatusra, etc.)
        - filename: Output filename (or None for auto-generated)
        
        Returns:
        - Filename of the generated MIDI file
        """
        # Create the tala
        tala = Tala(tala_name)
        
        # Generate patterns with lo-fi friendly variations
        patterns = self.tala_generator.generate_tala_pattern(
            tala, nadai=nadai, variations=True, length_multiplier=cycles
        )
        
        # For lo-fi, we want to emphasize the drum kit elements and perhaps add
        # some additional patterns common in lo-fi (like consistent hi-hats)
        self._adapt_for_lo_fi(patterns, tala.get_length() * nadai * cycles)
        
        # Create MIDI file
        if not filename:
            timestamp = int(time.time())
            filename = f"outputs/lofi_{tala_name}_rhythm_{timestamp}.mid"
        
        return self.tala_generator.create_midi_from_patterns(patterns, filename, bpm)
    
    def _adapt_for_lo_fi(self, patterns, total_length):
        """
        Modify patterns to be more lo-fi friendly.
        
        Parameters:
        - patterns: Dictionary of percussion patterns
        - total_length: Total length of the patterns
        """
        # If using Western kit, add consistent lo-fi elements
        if self.tala_generator.use_western_kit:
            # Common lo-fi beat elements:
            
            # 1. Consistent kick pattern (usually on beats 1 and 3 in 4/4)
            if "kick" in patterns:
                self._make_kick_lo_fi(patterns["kick"], total_length)
            
            # 2. Laid-back snare (usually on beats 2 and 4 in 4/4)
            if "snare" in patterns:
                self._make_snare_lo_fi(patterns["snare"], total_length)
            
            # 3. Consistent hi-hat pattern with subtle variations
            if "hihat" in patterns:
                self._make_hihat_lo_fi(patterns["hihat"], total_length)
                
            # 4. Occasional open hi-hat
            if "hihat_open" in patterns:
                self._add_lo_fi_open_hihat(patterns["hihat_open"], patterns["hihat"], total_length)
        else:
            # For Indian percussion, make patterns more consistent and laid-back
            if "mridangam_right" in patterns:
                self._make_mridangam_lo_fi(patterns["mridangam_right"], total_length)
            
            if "kanjira" in patterns:
                self._make_kanjira_lo_fi(patterns["kanjira"], total_length)
            
            # Add some consistent patterns for accompaniment feel
            if "ghatam" in patterns:
                self._make_ghatam_lo_fi(patterns["ghatam"], total_length)
    
    def _make_kick_lo_fi(self, kick_pattern, total_length):
        """Make kick pattern more lo-fi friendly."""
        # Lo-fi kicks are typically consistent but with subtle variations in velocity
        new_pattern = [0] * total_length
        
        # Place kicks on primary beats with slight velocity variation
        for i in range(0, total_length, 8):  # Assuming 4/4 with 16th note subdivision
            if i < total_length:
                new_pattern[i] = 100 + random.randint(-10, 5)  # Beat 1
            
            if i + 4 < total_length:
                new_pattern[i + 4] = 90 + random.randint(-10, 5)  # Beat 3
            
            # Occasionally add an off-beat kick for variation
            if random.random() < 0.2 and i + 6 < total_length:
                new_pattern[i + 6] = 80 + random.randint(-10, 5)
        
        # Preserve some of the original pattern for authenticity
        for i in range(total_length):
            if kick_pattern[i] > 0 and random.random() < 0.4:
                new_pattern[i] = kick_pattern[i]
        
        # Update the pattern
        for i in range(total_length):
            kick_pattern[i] = new_pattern[i]
    
    def _make_snare_lo_fi(self, snare_pattern, total_length):
        """Make snare pattern more lo-fi friendly."""
        new_pattern = [0] * total_length
        
        # Place snares on beats 2 and 4 with laid-back feel
        for i in range(0, total_length, 8):  # Assuming 4/4 with 16th note subdivision
            # Beat 2
            if i + 2 < total_length:
                new_pattern[i + 2] = 85 + random.randint(-8, 8)
            
            # Beat 4
            if i + 6 < total_length:
                new_pattern[i + 6] = 90 + random.randint(-8, 8)
            
            # Occasionally add a ghost note
            if random.random() < 0.3:
                ghost_pos = i + random.choice([1, 3, 5, 7])
                if ghost_pos < total_length:
                    new_pattern[ghost_pos] = 50 + random.randint(-10, 10)
        
        # Preserve some of the original pattern for authenticity
        for i in range(total_length):
            if snare_pattern[i] > 0 and random.random() < 0.3:
                new_pattern[i] = snare_pattern[i]
        
        # Update the pattern
        for i in range(total_length):
            snare_pattern[i] = new_pattern[i]
    
    def _make_hihat_lo_fi(self, hihat_pattern, total_length):
        """Make hi-hat pattern more lo-fi friendly."""
        new_pattern = [0] * total_length
        
        # Lo-fi typically has consistent 8th or 16th note hi-hats with velocity variations
        for i in range(total_length):
            # Every 8th note gets a hi-hat
            if i % 2 == 0:
                # Slight accent on main beats
                if i % 8 == 0:
                    new_pattern[i] = 75 + random.randint(-5, 5)
                else:
                    new_pattern[i] = 65 + random.randint(-8, 8)
            # For 16th note pattern, add softer hi-hats in between
            elif random.random() < 0.7:  # 70% chance for 16th notes
                new_pattern[i] = 50 + random.randint(-10, 10)
        
        # Add subtle swing feel (delay some hi-hats slightly)
        # This would be handled in the MIDI timing, not the pattern itself
        
        # Update the pattern
        for i in range(total_length):
            hihat_pattern[i] = new_pattern[i]
    
    def _add_lo_fi_open_hihat(self, open_hihat_pattern, closed_hihat_pattern, total_length):
        """Add occasional open hi-hats for lo-fi feel."""
        # Clear existing pattern first
        for i in range(total_length):
            open_hihat_pattern[i] = 0
        
        # Add open hi-hats at specific points (typically end of 2-bar phrases)
        for i in range(7, total_length, 16):  # End of each 2-bar phrase
            if random.random() < 0.6:  # 60% chance
                open_hihat_pattern[i] = 75 + random.randint(-5, 10)
                # Remove closed hi-hat at this position
                if i < len(closed_hihat_pattern):
                    closed_hihat_pattern[i] = 0
        
        # Occasionally add open hi-hats elsewhere for variation
        for i in range(3, total_length, 8):  # Off-beat positions
            if random.random() < 0.2:  # 20% chance
                open_hihat_pattern[i] = 70 + random.randint(-5, 10)
                # Remove closed hi-hat at this position
                if i < len(closed_hihat_pattern):
                    closed_hihat_pattern[i] = 0
    
    def _make_mridangam_lo_fi(self, mridangam_pattern, total_length):
        """Adapt mridangam pattern for lo-fi feel."""
        new_pattern = [0] * total_length
        
        # Create a more consistent pattern similar to lo-fi kick/snare
        for i in range(0, total_length, 8):  # Assuming 4/4 with 16th note subdivision
            if i < total_length:
                new_pattern[i] = 100 + random.randint(-10, 5)  # Beat 1
            
            if i + 2 < total_length:
                new_pattern[i + 2] = 85 + random.randint(-8, 8)  # Beat 2
            
            if i + 4 < total_length:
                new_pattern[i + 4] = 90 + random.randint(-10, 5)  # Beat 3
            
            if i + 6 < total_length:
                new_pattern[i + 6] = 88 + random.randint(-8, 8)  # Beat 4
        
        # Preserve some of the original pattern for authenticity
        for i in range(total_length):
            if mridangam_pattern[i] > 0 and random.random() < 0.4:
                new_pattern[i] = mridangam_pattern[i]
        
        # Update the pattern
        for i in range(total_length):
            mridangam_pattern[i] = new_pattern[i]
    
    def _make_kanjira_lo_fi(self, kanjira_pattern, total_length):
        """Adapt kanjira pattern for lo-fi feel."""
        new_pattern = [0] * total_length
        
        # Similar to hi-hat pattern but with more variation
        for i in range(total_length):
            # Every 8th note
            if i % 2 == 0:
                # Accents on certain beats
                if i % 8 in [2, 6]:  # Beats 2 and 4
                    new_pattern[i] = 80 + random.randint(-8, 8)
                elif i % 8 == 0:  # Beat 1
                    new_pattern[i] = 70 + random.randint(-10, 5)
                else:
                    new_pattern[i] = 65 + random.randint(-10, 10)
        
        # Preserve some of the original pattern
        for i in range(total_length):
            if kanjira_pattern[i] > 0 and random.random() < 0.3:
                new_pattern[i] = kanjira_pattern[i]
        
        # Update the pattern
        for i in range(total_length):
            kanjira_pattern[i] = new_pattern[i]
    
    def _make_ghatam_lo_fi(self, ghatam_pattern, total_length):
        """Adapt ghatam pattern for lo-fi feel."""
        new_pattern = [0] * total_length
        
        # Use ghatam for occasional accents
        for i in range(0, total_length, 8):
            if random.random() < 0.4:  # 40% chance on main beats
                accent_pos = i + random.choice([0, 4])  # Beats 1 or 3
                if accent_pos < total_length:
                    new_pattern[accent_pos] = 85 + random.randint(-10, 10)
            
            if random.random() < 0.25:  # 25% chance on off-beats
                accent_pos = i + random.choice([2, 6])  # Beats 2 or 4
                if accent_pos < total_length:
                    new_pattern[accent_pos] = 80 + random.randint(-10, 10)
        
        # Preserve some of the original pattern
        for i in range(total_length):
            if ghatam_pattern[i] > 0 and random.random() < 0.3:
                new_pattern[i] = ghatam_pattern[i]
        
        # Update the pattern
        for i in range(total_length):
            ghatam_pattern[i] = new_pattern[i]


# Integration function to connect with the main generator
def create_carnatic_rhythm(bpm=75, tala_name="adi", nadai=4, cycles=4, use_western_kit=True, filename=None):
    """
    Create a Carnatic rhythm pattern suitable for lo-fi production.
    
    Parameters:
    - bpm: Tempo in beats per minute
    - tala_name: Name of the tala to use
    - nadai: Subdivision level (3=tisra, 4=chatusra, etc.)
    - cycles: Number of tala cycles to generate
    - use_western_kit: Whether to use Western drum sounds instead of Indian percussion
    - filename: Output filename (or None for auto-generated)
    
    Returns:
    - Filename of the generated MIDI file
    """
    adapter = TalaLoFiAdapter(use_western_kit)
    return adapter.generate_lo_fi_rhythm(tala_name, cycles, bpm, nadai, filename)


# Example usage
if __name__ == "__main__":
    # Create a simple Adi tala pattern
    rhythm_file = create_carnatic_rhythm(
        bpm=75, 
        tala_name="adi",  # 8-beat cycle
        nadai=4,  # Chatusra nadai (4 subdivisions per beat)
        cycles=4,  # Generate 4 cycles of the tala
        use_western_kit=True  # Use Western drum kit for lo-fi feel
    )
    
    print(f"Generated rhythm file: {rhythm_file}")
    
    # List available talas
    print("\nAvailable Carnatic talas:")
    for tala_name in Tala.COMMON_TALAS.keys():
        tala = Tala(tala_name)
        print(f"- {tala}")