#!/usr/bin/env python3
"""
Enhanced Raga-based Lo-Fi Melody Generator
------------------------------------------
This module creates authentic raga-based melodies suitable for lo-fi music production.
It uses traditional Indian raga structures and patterns to generate MIDI files.
"""

import os
import json
import random
import mido
from mido import Message, MidiFile, MidiTrack
import time

class EnhancedRagaGenerator:
    """Generate melodies and patterns based on Indian ragas."""
    
    def __init__(self, ragas_file='data/ragas.json'):
        """Initialize the generator with raga data from JSON file."""
        # Load raga data
        try:
            with open(ragas_file, 'r') as f:
                data = json.load(f)
                self.ragas_data = {raga['id']: raga for raga in data['ragas']}
                self.time_categories = data.get('time_categories', {})
                self.mood_categories = data.get('mood_categories', {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading ragas file: {e}")
            # Initialize with empty data as fallback
            self.ragas_data = {}
            self.time_categories = {}
            self.mood_categories = {}
    
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
                'suitable_for': raga['suitable_for']
            }
            for raga_id, raga in self.ragas_data.items()
        ]

    def generate_melody(self, raga_id, length=16, use_patterns=True, base_note=60, bpm=75):
        """
        Generate a melody based on a specific raga.
        
        Parameters:
        - raga_id: ID of the raga to use
        - length: Number of notes in the melody
        - use_patterns: Whether to use characteristic patterns from the raga
        - base_note: Base MIDI note for Sa (default: 60 = middle C)
        - bpm: Tempo in beats per minute
        
        Returns:
        - Filename of the generated MIDI file
        """
        if raga_id not in self.ragas_data:
            raise ValueError(f"Raga {raga_id} not found")
            
        raga = self.ragas_data[raga_id]
        notes = []
        
        # If using patterns, build melody from existing patterns
        if use_patterns and 'characteristic_phrases' in raga:
            patterns = raga['characteristic_phrases'] + raga.get('common_patterns', [])
            
            # Generate melody by combining patterns
            remaining_length = length
            while remaining_length > 0:
                # Choose a random pattern that fits
                suitable_patterns = [p for p in patterns if len(p) <= remaining_length]
                if not suitable_patterns:
                    # If no pattern fits, use single notes from the scale
                    notes.append(random.choice(raga['arohan']))
                    remaining_length -= 1
                else:
                    pattern = random.choice(suitable_patterns)
                    notes.extend(pattern)
                    remaining_length -= len(pattern)
        else:
            # Use more traditional algorithmic approach
            current_note_idx = 0  # Start with Sa
            
            for i in range(length):
                # Decide whether to use ascending or descending scale
                if i < length * 0.6:  # First 60% of melody tends to ascend
                    scale = raga['arohan']
                    
                    # Determine next note index (with tendency to move upward)
                    if current_note_idx < len(scale) - 1:
                        move_prob = random.random()
                        if move_prob < 0.7:  # 70% chance to move up
                            current_note_idx += 1
                        elif move_prob < 0.9:  # 20% chance to stay
                            pass
                        else:  # 10% chance to move down
                            current_note_idx = max(0, current_note_idx - 1)
                    else:
                        # At top of scale, start moving down
                        current_note_idx = len(scale) - 2
                else:
                    # Last 40% of melody tends to descend
                    scale = raga['avarohan']
                    
                    # Map current note to descending scale
                    if i == int(length * 0.6):  # First note of descent
                        # Find current note in descending scale to maintain continuity
                        current_value = raga['arohan'][current_note_idx]
                        if current_value in raga['avarohan']:
                            current_note_idx = raga['avarohan'].index(current_value)
                        else:
                            # If note not in avarohan, start from top
                            current_note_idx = 0
                    
                    # Determine next note index (with tendency to move downward)
                    if current_note_idx < len(scale) - 1:
                        move_prob = random.random()
                        if move_prob < 0.7:  # 70% chance to move down
                            current_note_idx += 1
                        elif move_prob < 0.9:  # 20% chance to stay
                            pass
                        else:  # 10% chance to move up
                            current_note_idx = max(0, current_note_idx - 1)
                    else:
                        # At bottom of scale, we can wrap to top or stay
                        if random.random() < 0.3:  # 30% chance to cycle
                            current_note_idx = 0
                
                # Add note to sequence
                notes.append(scale[current_note_idx])
                
        # Emphasize vadi and samvadi notes (important notes in raga)
        for i in range(1, len(notes)-1):
            # 20% chance to replace a note with vadi or samvadi
            if random.random() < 0.2:
                if random.random() < 0.7:  # 70% for vadi
                    notes[i] = raga['vadi']
                else:  # 30% for samvadi
                    notes[i] = raga['samvadi']
        
        # Create MIDI file
        timestamp = int(time.time())
        filename = f"outputs/{raga_id}_lofi_melody_{timestamp}.mid"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self._create_midi(notes, filename, raga['name'], base_note, bpm)
        return filename
    
    def generate_chord_progression(self, raga_id, length=4, base_note=48, bpm=75):
        """
        Generate a chord progression suitable for the selected raga.
        
        Parameters:
        - raga_id: ID of the raga to use
        - length: Number of chords in the progression
        - base_note: Base MIDI note for Sa (default: 48 = C3)
        - bpm: Tempo in beats per minute
        
        Returns:
        - Filename of the generated MIDI file
        """
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
        """
        Generate a simple bass line based on the raga.
        
        Parameters:
        - raga_id: ID of the raga to use
        - length: Number of notes in the bass line
        - base_note: Base MIDI note for Sa (default: 36 = C2)
        - bpm: Tempo in beats per minute
        
        Returns:
        - Filename of the generated MIDI file
        """
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
            if random.random() < 0.3:  # 30% chance for no third
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