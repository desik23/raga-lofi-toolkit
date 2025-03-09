#!/usr/bin/env python3
"""
Enhanced Raga-based Lo-Fi Melody Generator
------------------------------------------
This module creates authentic raga-based melodies suitable for lo-fi music production.
It uses traditional Indian raga structures and patterns to generate MIDI files 
with support for both Hindustani and Carnatic traditions.
"""

import os
import json
import random
import mido
from mido import Message, MidiFile, MidiTrack
import time

class CarnaticFeatures:
    """Helper class for Carnatic-specific music generation features."""
    
    def __init__(self, generator):
        """Initialize with reference to the main generator."""
        self.generator = generator
        self.setup_carnatic_mappings()
    
    def setup_carnatic_mappings(self):
        """Set up mappings between Carnatic and Hindustani systems."""
        try:
            self.c_to_h = self.generator.ragas_data.get('system_mapping', {}).get('carnatic_to_hindustani', {})
            self.h_to_c = self.generator.ragas_data.get('system_mapping', {}).get('hindustani_to_carnatic', {})
        except (KeyError, AttributeError):
            # Default to empty mappings if not available
            self.c_to_h = {}
            self.h_to_c = {}
    
    def get_equivalent_raga(self, raga_id, target_system='hindustani'):
        """Get the equivalent raga ID in the target system."""
        if target_system == 'hindustani':
            return self.c_to_h.get(raga_id, None)
        else:  # carnatic
            return self.h_to_c.get(raga_id, None)
    
    def get_ragas_by_melakarta(self, melakarta_number):
        """Return list of raga IDs that belong to a specific melakarta number."""
        return [
            raga_id for raga_id, raga in self.generator.ragas_data.items()
            if raga.get('melakarta') == melakarta_number
        ]
    
    def get_carnatic_ragas(self):
        """Return list of all Carnatic raga IDs."""
        return [
            raga_id for raga_id, raga in self.generator.ragas_data.items()
            if raga.get('system') == 'carnatic'
        ]
    
    def get_hindustani_ragas(self):
        """Return list of all Hindustani raga IDs."""
        return [
            raga_id for raga_id, raga in self.generator.ragas_data.items()
            if raga.get('system') == 'hindustani' or raga.get('system') is None  # For backward compatibility
        ]
    
    def apply_gamaka(self, notes, raga_id):
        """
        Apply characteristic Carnatic gamaka ornamentations to a note sequence.
        """
        if raga_id not in self.generator.ragas_data:
            return notes
            
        raga = self.generator.ragas_data[raga_id]
        if raga.get('system') != 'carnatic':
            return notes
            
        # Apply generic approach since we're missing specific implementations
        return self._generic_gamakas(notes, raga)
    
    def _generic_gamakas(self, notes, raga):
        """Apply generic gamaka patterns based on raga structure."""
        result = []
        
        for i, note in enumerate(notes):
            # 30% chance to apply a gamaka if not the last note
            if i < len(notes) - 1 and random.random() < 0.3:
                next_note = notes[i+1]
                
                # If moving up, add a brief oscillation
                if next_note > note:
                    result.append(note)
                    # Brief touch of a note in between
                    if note + 1 in raga['arohan']:
                        result.append(note + 0.5)  # Using 0.5 to indicate a quick touch
                
                # If moving down, add a slide
                elif next_note < note:
                    result.append(note)
                    # Slide down
                    step = -0.5 if note - next_note > 2 else -0.25
                    current = note
                    while current > next_note + 0.5:
                        current += step
                        result.append(current)
                
                # If same note, add a slight oscillation
                else:
                    result.append(note)
                    # Subtle oscillation around the note
                    result.append(note + 0.25)
                    result.append(note - 0.25)
            else:
                result.append(note)
        
        return result
    
    def apply_gamaka_with_pitch_bend(self, notes, raga_id, intensity=1.0):
        """
        Apply characteristic Carnatic gamaka ornamentations to notes using pitch bends.
        """
        if intensity <= 0.1:
            # Return notes without gamakas if intensity is nearly zero
            return [{'note': note, 'bend': None} for note in notes]
        
        if raga_id not in self.generator.ragas_data:
            return [{'note': note, 'bend': None} for note in notes]
            
        raga = self.generator.ragas_data[raga_id]
        if raga.get('system') != 'carnatic':
            return [{'note': note, 'bend': None} for note in notes]
        
        # Use generic approach instead of specific patterns
        return self._generic_pitch_bends(notes, raga, intensity)
    
    def _generic_pitch_bends(self, notes, raga, intensity=1.0):
        """Apply generic gamaka patterns using pitch bends based on raga structure."""
        result = []
        for i, note in enumerate(notes):
            # Basic note with no bend initially
            note_event = {'note': note, 'bend': None}
            
            # Skip last note for forward-looking patterns
            if i < len(notes) - 1:
                next_note = notes[i+1]
                
                # Probability of applying gamaka based on intensity
                if random.random() < 0.3 * intensity:
                    # Moving upward: add oscillation
                    if next_note > note:
                        # Create pitch bend data
                        bend_points = []
                        
                        # Starting at current note
                        bend_points.append((0, 0))  # (time_offset_percentage, pitch_offset_semitones)
                        
                        # Small oscillation up and down
                        oscillation_size = 0.3 * intensity
                        bend_points.append((0.3, oscillation_size))
                        bend_points.append((0.6, -oscillation_size))
                        bend_points.append((1.0, 0))
                        
                        note_event['bend'] = bend_points
                    
                    # Moving downward: add slide
                    elif next_note < note:
                        # Create slide effect
                        bend_points = []
                        bend_points.append((0, 0))
                        
                        # Gradual slide down toward the next note
                        difference = note - next_note
                        if difference > 1:
                            # For larger intervals, add a characteristic slide
                            slide_amount = min(difference * 0.6 * intensity, 2.0)
                            bend_points.append((0.4, 0))
                            bend_points.append((0.7, -slide_amount))
                            bend_points.append((1.0, 0))
                        else:
                            # For smaller intervals, gentler slide
                            bend_points.append((0.5, -0.3 * intensity))
                            bend_points.append((1.0, 0))
                        
                        note_event['bend'] = bend_points
                    
                    # Same note: add slight oscillation for emphasis
                    else:
                        # Oscillate around the pitch
                        bend_points = []
                        bend_points.append((0, 0))
                        bend_points.append((0.3, 0.2 * intensity))
                        bend_points.append((0.6, -0.2 * intensity))
                        bend_points.append((1.0, 0))
                        
                        note_event['bend'] = bend_points
            
            result.append(note_event)
        
        return result
        
    def create_midi_with_pitch_bends(self, note_events, filename, track_name, base_note=60, bpm=75):
        """
        Create a MIDI file from note events that include pitch bend data.
        """
        midi = MidiFile()
        track = MidiTrack()
        midi.tracks.append(track)
        
        # Add tempo
        tempo = mido.bpm2tempo(bpm)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))
        
        # Add track name
        track.append(mido.MetaMessage('track_name', name=track_name))
        
        # Add time signature (4/4 for lo-fi)
        track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))
        
        # Determine duration (eighth notes)
        ticks_per_beat = midi.ticks_per_beat
        duration = ticks_per_beat // 2
        
        # MIDI uses a pitch bend range of -8192 to +8191
        # We'll scale our semitone-based bends to this range
        # Standard pitch bend range is ±2 semitones (but this can vary by synth)
        BEND_RANGE = 2  # semitones
        MAX_BEND = 8191
        
        # Process each note
        prev_note = None
        for i, note_event in enumerate(note_events):
            note_value = note_event['note']
            bend_data = note_event['bend']
            
            # Calculate MIDI note number
            midi_note = base_note + int(note_value)  # Only use the integer part for note number
            
            # Humanize velocity and timing
            velocity = random.randint(70, 90)
            time_variation = random.randint(-5, 5)
            
            # Note on event
            track.append(Message('note_on', note=midi_note, velocity=velocity, 
                            time=max(0, time_variation)))
            
            # Apply pitch bends if specified
            if bend_data:
                prev_time = 0
                
                for time_pct, bend_amt in bend_data:
                    # Calculate when this bend should happen
                    time_ticks = int(time_pct * duration)
                    time_delta = time_ticks - prev_time
                    prev_time = time_ticks
                    
                    # Calculate bend value
                    bend_value = int((bend_amt / BEND_RANGE) * MAX_BEND)
                    
                    # Add pitch bend message
                    track.append(Message('pitchwheel', pitch=bend_value, time=max(0, time_delta)))
            
            # Reset pitch bend at end of note
            if bend_data:
                track.append(Message('pitchwheel', pitch=0, time=0))
            
            # Note off event
            note_duration = duration + random.randint(-10, 10)
            track.append(Message('note_off', note=midi_note, velocity=0, 
                            time=max(1, note_duration)))
        
        # Save MIDI file
        midi.save(filename)
        return filename    


class EnhancedRagaGenerator:
    """Generate melodies and patterns based on Indian ragas."""
    
    def __init__(self, ragas_file='data/ragas.json'):
        """Initialize the generator with raga data from JSON file."""
        # Load raga data
        try:
            with open(ragas_file, 'r') as f:
                data = json.load(f)
                # First setup the ragas data
                self.ragas_data = {raga['id']: raga for raga in data['ragas']}
                self.time_categories = data.get('time_categories', {})
                self.mood_categories = data.get('mood_categories', {})
                # Then add Carnatic support
                self.carnatic = CarnaticFeatures(self)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading ragas file: {e}")
            # Initialize with empty data as fallback
            self.ragas_data = {}
            self.time_categories = {}
            self.mood_categories = {}
            # Still create carnatic object with empty data
            self.carnatic = CarnaticFeatures(self)
    
    def get_ragas_by_mood(self, mood):
        """Return list of raga IDs suitable for a specific mood."""
        return self.mood_categories.get(mood, [])
    
    def get_ragas_by_time(self, time_of_day):
        """Return list of raga IDs suitable for a specific time of day."""
        return self.time_categories.get(time_of_day, [])
    
    def list_available_ragas(self):
        """Return a list of all available ragas with their details."""
        return [
            {
                'id': raga_id,
                'name': raga['name'],
                'mood': raga['mood'],
                'time': raga['time'],
                'suitable_for': raga['suitable_for'],
                'system': raga.get('system', 'hindustani')  # Add system for UI filtering
            }
            for raga_id, raga in self.ragas_data.items()
        ]

    def generate_melody(self, raga_id, length=16, use_patterns=True, base_note=60, bpm=75, gamaka_intensity=1.0, strict_rules=False):
        """
        Generate a melody based on a specific raga with Carnatic support.
        """
        if raga_id not in self.ragas_data:
            raise ValueError(f"Raga {raga_id} not found")
            
        raga = self.ragas_data[raga_id]
        notes = []
        
        # Simplified melody generation - use existing logic
        if use_patterns and 'characteristic_phrases' in raga:
            patterns = raga['characteristic_phrases']
            if 'common_patterns' in raga:
                patterns.extend(raga['common_patterns'])
            
            # Generate melody by combining patterns
            remaining_length = length
            while remaining_length > 0:
                suitable_patterns = [p for p in patterns if len(p) <= remaining_length]
                if not suitable_patterns:
                    if remaining_length > len(raga['arohan']) // 2:
                        phrase = raga['arohan'][:remaining_length]
                    else:
                        phrase = raga['avarohan'][-remaining_length:]
                    notes.extend(phrase)
                    remaining_length -= len(phrase)
                else:
                    pattern = random.choice(suitable_patterns)
                    notes.extend(pattern)
                    remaining_length -= len(pattern)
        else:
            # Use traditional arohan/avarohan patterns
            arohan_len = min(length // 2, len(raga['arohan']))
            notes.extend(raga['arohan'][:arohan_len])
            
            remaining = length - arohan_len
            if remaining > 0:
                avarohan_len = min(remaining, len(raga['avarohan']))
                notes.extend(raga['avarohan'][:avarohan_len])
        
        # Check if this is a Carnatic raga and apply gamakas if appropriate
        if raga.get('system') == 'carnatic' and hasattr(self, 'carnatic'):
            if gamaka_intensity > 0.0:
                # Apply gamakas with specified intensity
                note_events = self.carnatic.apply_gamaka_with_pitch_bend(notes, raga_id, gamaka_intensity)
                
                # Create MIDI file with pitch bends
                timestamp = int(time.time())
                filename = f"outputs/{raga_id}_lofi_melody_{timestamp}.mid"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                return self.carnatic.create_midi_with_pitch_bends(
                    note_events, filename, raga['name'], base_note, bpm
                )
        
        # For non-Carnatic ragas or when gamaka intensity is zero, use regular MIDI creation
        timestamp = int(time.time())
        filename = f"outputs/{raga_id}_lofi_melody_{timestamp}.mid"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self._create_midi(notes, filename, raga['name'], base_note, bpm)
        return filename
    
    def generate_chord_progression(self, raga_id, length=4, base_note=48, bpm=75, strict_rules=False):
        """Generate a chord progression suitable for the selected raga."""
        if raga_id not in self.ragas_data:
            raise ValueError(f"Raga {raga_id} not found")
            
        raga = self.ragas_data[raga_id]
        scale = raga['arohan']
        
        # Define common chord progressions for lo-fi based on raga type
        # Using scale degrees (0-based indexing)
        major_progressions = [
            [0, 5, 3, 4],  # I-vi-IV-V
            [0, 4, 5, 3],  # I-V-vi-IV
            [0, 3, 4, 0],  # I-IV-V-I
            [0, 3, 0, 4]   # I-IV-I-V
        ]
        
        minor_progressions = [
            [0, 3, 4, 0],  # i-iv-v-i
            [0, 5, 3, 4],  # i-VI-iv-v
            [0, 5, 0, 4],  # i-VI-i-v
            [0, 3, 5, 4]   # i-iv-VI-v
        ]
        
        # Determine if raga has major or minor feel
        # In Indian classical, this is complex, but we'll simplify for lo-fi context
        # Check if third scale degree (Ga) is major (4 semitones from Sa) or minor (3 semitones)
        if 4 in scale and scale.index(4) == 2:  # Major third
            chord_patterns = major_progressions
        else:  # Minor third
            chord_patterns = minor_progressions
        
        # Select a chord progression
        chord_progression = random.choice(chord_patterns)
        
        # Repeat progression to reach desired length
        while len(chord_progression) < length:
            chord_progression.extend(chord_progression[:length-len(chord_progression)])
        
        # Create MIDI file
        timestamp = int(time.time())
        filename = f"outputs/{raga_id}_lofi_chords_{timestamp}.mid"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self._create_chord_midi(chord_progression, scale, filename, raga['name'], base_note, bpm)
        return filename
    
    def generate_bass_line(self, raga_id, length=16, base_note=36, bpm=75):
        """Generate a simple bass line based on the raga."""
        if raga_id not in self.ragas_data:
            raise ValueError(f"Raga {raga_id} not found")
            
        raga = self.ragas_data[raga_id]
        
        # Bass lines typically use just the main notes of the scale
        # For lo-fi, we'll mainly use Sa, Pa (perfect fifth), and occasionally Ma (fourth)
        bass_notes = []
        
        # Find the indices of Sa, Ma, Pa in the scale
        sa_index = 0  # Sa is always the first note
        pa_index = raga['arohan'].index(7) if 7 in raga['arohan'] else None
        ma_index = raga['arohan'].index(5) if 5 in raga['arohan'] else None
        
        # Generate a simple pattern with emphasis on strong beats
        for i in range(length):
            if i % 4 == 0:  # Strong beat (first of bar) - usually Sa
                bass_notes.append(raga['arohan'][sa_index])
            elif i % 4 == 2:  # Strong beat (third of bar) - usually Pa or Ma
                if pa_index is not None and random.random() < 0.7:
                    bass_notes.append(raga['arohan'][pa_index])
                elif ma_index is not None:
                    bass_notes.append(raga['arohan'][ma_index])
                else:
                    bass_notes.append(raga['arohan'][sa_index])
            else:  # Weak beats - either repeat previous or rest
                if random.random() < 0.6:  # 60% chance for note
                    bass_notes.append(bass_notes[-1])
                else:  # 40% chance for rest
                    bass_notes.append(-1)  # Use -1 to indicate rest
        
        # Create MIDI file
        timestamp = int(time.time())
        filename = f"outputs/{raga_id}_lofi_bass_{timestamp}.mid"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self._create_midi(bass_notes, filename, f"{raga['name']} Bass", base_note, bpm, is_bass=True)
        return filename
    
    def _create_midi(self, notes, filename, track_name, base_note=60, bpm=75, is_bass=False):
        """Create a MIDI file from the note sequence."""
        midi = MidiFile()
        track = MidiTrack()
        midi.tracks.append(track)
        
        # Add tempo
        tempo = mido.bpm2tempo(bpm)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))
        
        # Add track name
        track.append(mido.MetaMessage('track_name', name=track_name))
        
        # Add time signature (4/4 for lo-fi)
        track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))
        
        # Determine duration (eighth notes for melody, quarter notes for bass)
        ticks_per_beat = midi.ticks_per_beat
        duration = ticks_per_beat // 2 if not is_bass else ticks_per_beat
        
        # Create humanized velocity for lo-fi feel
        def get_humanized_velocity():
            # Lo-fi typically has a soft, consistent velocity with subtle variations
            base_velocity = 70 if not is_bass else 80
            variation = random.randint(-10, 10)
            return max(40, min(100, base_velocity + variation))
        
        # Add notes
        time = 0
        for note_value in notes:
            if note_value == -1:  # Rest
                track.append(Message('note_off', note=0, velocity=0, time=duration))
                continue
                
            velocity = get_humanized_velocity()
            
            # Calculate MIDI note number
            midi_note = base_note + note_value
            
            # Keep note in reasonable range
            while midi_note > 100:
                midi_note -= 12
            while midi_note < 30:
                midi_note += 12
            
            # Add note with slight timing variations for human feel
            time_variation = random.randint(-5, 5) if not is_bass else 0
            
            # Note on
            track.append(Message('note_on', note=midi_note, velocity=velocity, 
                                time=max(0, time_variation)))
            
            # Note off
            note_duration = duration + random.randint(-10, 10) if not is_bass else duration
            track.append(Message('note_off', note=midi_note, velocity=0, 
                                time=max(1, note_duration)))
            
        # Save MIDI file
        midi.save(filename)
        return filename
    
    def _create_chord_midi(self, chord_progression, scale, filename, track_name, base_note=48, bpm=75):
        """Create a MIDI file with chord progression."""
        midi = MidiFile()
        track = MidiTrack()
        midi.tracks.append(track)
        
        # Add tempo
        tempo = mido.bpm2tempo(bpm)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))
        
        # Add track name
        track.append(mido.MetaMessage('track_name', name=f"{track_name} Chords"))
        
        # Add time signature
        track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))
        
        # Duration for one bar (4 beats)
        ticks_per_beat = midi.ticks_per_beat
        bar_duration = ticks_per_beat * 4
        
        # Function to build a chord from a root note
        def build_chord(root_idx):
            chord_notes = []
            
            # Add root note
            root_note = scale[root_idx % len(scale)]
            chord_notes.append(base_note + root_note)
            
            # Try to add third (if within scale)
            third_idx = (root_idx + 2) % len(scale)
            if third_idx < len(scale):
                chord_notes.append(base_note + scale[third_idx])
            
            # Try to add fifth (if within scale)
            fifth_idx = (root_idx + 4) % len(scale)
            if fifth_idx < len(scale):
                chord_notes.append(base_note + scale[fifth_idx])
            
            return chord_notes
        
        # Create chords
        for chord_root in chord_progression:
            # Build chord
            chord_notes = build_chord(chord_root)
            
            # For lo-fi feel, sometimes use voicings without the third
            if random.random() < 0.3 and len(chord_notes) > 2:  # 30% chance for no third
                chord_notes = [note for i, note in enumerate(chord_notes) if i != 1]
            
            # Add some velocity variation for human feel
            velocity = random.randint(55, 75)  # Soft velocity for lo-fi
            
            # Note on events
            for i, note in enumerate(chord_notes):
                # First note has no delay, others are simultaneous
                track.append(Message('note_on', note=note, velocity=velocity, 
                                    time=0 if i > 0 else 0))
            
            # Note off events (after one bar)
            for i, note in enumerate(chord_notes):
                # Last note gets the full duration, others get 0
                off_time = bar_duration if i == len(chord_notes) - 1 else 0
                track.append(Message('note_off', note=note, velocity=0, time=off_time))
        
        # Save MIDI file
        midi.save(filename)
        return filename
    
    def generate_complete_track(self, raga_id, base_note=60, bpm=75):
        """
        Generate a complete set of MIDI files for a lo-fi track.
        
        Returns:
        - Dictionary with filenames for each component
        """
        melody = self.generate_melody(raga_id, length=32, base_note=base_note, bpm=bpm)
        chords = self.generate_chord_progression(raga_id, length=4, base_note=base_note-12, bpm=bpm)
        bass = self.generate_bass_line(raga_id, length=32, base_note=base_note-24, bpm=bpm)
        
        return {
            'melody': melody,
            'chords': chords,
            'bass': bass,
            'raga': self.ragas_data[raga_id]['name'],
            'bpm': bpm
        }
    
    def get_melakarta_info(self, number):
        """Get information about a melakarta raga."""
        # Fixed to check if 'melakarta_info' is a key in a dictionary
        if 'melakarta_info' in self.ragas_data and str(number) in self.ragas_data['melakarta_info']:
            return self.ragas_data['melakarta_info'][str(number)]
        return None

    def get_ragas_by_weather(self, weather):
        """Return list of raga IDs suitable for specific weather."""
        if 'weather_categories' in self.ragas_data and weather in self.ragas_data['weather_categories']:
            return self.ragas_data['weather_categories'][weather]
        return []


# Example usage
if __name__ == "__main__":
    generator = EnhancedRagaGenerator()
    
    # List available ragas
    print("Available Ragas:")
    for raga in generator.list_available_ragas():
        print(f"{raga['name']} - {raga['mood']} ({raga['time']})")
    
    # Generate a complete track
    print("\nGenerating track for Yaman raga...")
    track_files = generator.generate_complete_track('yaman', bpm=75)
    
    print(f"\nGenerated files:")
    for component, filename in track_files.items():
        if component != 'raga' and component != 'bpm':
            print(f"- {component}: {filename}")