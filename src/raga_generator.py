import random
import mido
from mido import Message, MidiFile, MidiTrack
import os

class RagaMelodyGenerator:
    def __init__(self):
        # Define common ragas with their notes (in scale degrees)
        # Using 0-based indexing where 0 = Sa, 1 = Komal Re, 2 = Re, etc.
        self.ragas = {
            "yaman": {
                "name": "Yaman",
                "arohan": [0, 2, 4, 6, 7, 9, 11, 12],  # Ascending pattern
                "avarohan": [12, 11, 9, 7, 6, 4, 2, 0],  # Descending pattern
                "vadi": 4,  # Ga
                "samvadi": 11,  # Ni
                "mood": "peaceful evening",
                "time": "evening"
            },
            "bhairav": {
                "name": "Bhairav",
                "arohan": [0, 1, 4, 5, 7, 8, 11, 12],
                "avarohan": [12, 11, 8, 7, 5, 4, 1, 0],
                "vadi": 1,  # Komal Re
                "samvadi": 8,  # Komal Dha
                "mood": "serious, profound",
                "time": "morning"
            },
            "malkauns": {
                "name": "Malkauns",
                "arohan": [0, 3, 5, 8, 10, 12],
                "avarohan": [12, 10, 8, 5, 3, 0],
                "vadi": 5,  # Ma
                "samvadi": 10,  # Komal Ni
                "mood": "deep, meditative",
                "time": "night"
            }
        }
        
    def generate_melody(self, raga_name, length=16, base_note=60, bpm=75):
        """Generate a melody based on a specific raga"""
        if raga_name not in self.ragas:
            raise ValueError(f"Raga {raga_name} not found")
            
        raga = self.ragas[raga_name]
        notes = []
        
        # Start with Sa
        current_position = 0
        
        # Lo-fi commonly uses simple patterns with repetition
        for i in range(length):
            # Decide if we're moving up or down (with more weight toward moving up initially)
            if i < length // 2:
                # First half tends to ascend
                if current_position < len(raga["arohan"]) - 1:
                    # 70% chance to move up in early part
                    if random.random() < 0.7:
                        current_position += 1
                    else:
                        # Sometimes stay on same note for emphasis
                        pass
                else:
                    # At top, start moving down
                    current_position = len(raga["arohan"]) - 2
            else:
                # Second half tends to descend
                if current_position > 0:
                    # 70% chance to move down in later part
                    if random.random() < 0.7:
                        current_position -= 1
                    else:
                        # Sometimes stay on same note
                        pass
                else:
                    # At bottom, may move up slightly
                    current_position = 1
            
            # Decide whether to use arohan or avarohan pattern
            if i < length // 2:
                # Use ascending pattern for first half
                scale = raga["arohan"]
            else:
                # Use descending pattern for second half
                current_position = len(raga["avarohan"]) - current_position - 1
                scale = raga["avarohan"]
            
            # Add note to sequence (ensuring we don't go out of bounds)
            idx = min(current_position, len(scale) - 1)
            notes.append(scale[idx])
        
        # Create MIDI file
        return self._create_midi(notes, raga_name, base_note, bpm)
    
    def _create_midi(self, notes, raga_name, base_note=60, bpm=75):
        """Create a MIDI file from a sequence of notes"""
        midi = MidiFile()
        track = MidiTrack()
        midi.tracks.append(track)
        
        # Add tempo
        tempo = mido.bpm2tempo(bpm)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))
        
        # Set time signature to 4/4
        track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))
        
        # Add notes (using eighth notes for lo-fi feel)
        ticks_per_beat = midi.ticks_per_beat
        duration = ticks_per_beat // 2  # Eighth notes
        
        time = 0
        for note in notes:
            # Add some velocity variation for more human feel
            velocity = random.randint(70, 90)
            
            # Map note value to actual MIDI note (adding base_note)
            midi_note = base_note + note
            
            # Note on
            track.append(Message('note_on', note=midi_note, velocity=velocity, time=0 if time == 0 else duration))
            
            # Note off
            track.append(Message('note_off', note=midi_note, velocity=0, time=duration))
            
            time += 1
        
        # Create filename with raga name
        filename = f"{raga_name}_lofi_melody.mid"
        midi.save(filename)
        return filename

    def generate_chord_progression(self, raga_name, length=4, base_note=60, bpm=75):
        """Generate a simple chord progression based on raga"""
        if raga_name not in self.ragas:
            raise ValueError(f"Raga {raga_name} not found")
            
        raga = self.ragas[raga_name]
        
        # For lo-fi, we'll use simple triads based on the raga's scale
        scale = raga["arohan"]
        
        # Common lo-fi progression pattern
        if raga_name == "yaman":
            # Yaman (major scale) - using I-vi-IV-V pattern
            chord_pattern = [0, 5, 3, 4]  # Using scale degrees as roots
        elif raga_name == "malkauns":
            # Minor scale pattern
            chord_pattern = [0, 3, 5, 2]
        else:
            # Generic pattern for other ragas
            chord_pattern = [0, 2, 3, 5]
        
        # Create MIDI file with chord progression
        return self._create_chord_midi(chord_pattern, scale, raga_name, base_note, bpm)
    
    def _create_chord_midi(self, chord_pattern, scale, raga_name, base_note=60, bpm=75):
        """Create a MIDI file with chord progression"""
        midi = MidiFile()
        track = MidiTrack()
        midi.tracks.append(track)
        
        # Add tempo
        tempo = mido.bpm2tempo(bpm)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))
        
        # Add time signature
        track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))
        
        # Calculate duration for one bar
        ticks_per_beat = midi.ticks_per_beat
        bar_duration = ticks_per_beat * 4  # 4 beats per bar
        
        # Create chords
        time = 0
        for chord_root in chord_pattern:
            # Build triad chord (1-3-5) if possible
            chord_notes = []
            
            # Root note
            root_idx = chord_root % len(scale)
            chord_notes.append(base_note + scale[root_idx])
            
            # Try to add third (if exists in scale)
            third_idx = (root_idx + 2) % len(scale)
            chord_notes.append(base_note + scale[third_idx])
            
            # Try to add fifth (if exists in scale)
            fifth_idx = (root_idx + 4) % len(scale)
            chord_notes.append(base_note + scale[fifth_idx])
            
            # Add chord notes
            velocity = 60  # Softer for chords
            
            # Note on for all chord notes
            for note in chord_notes:
                track.append(Message('note_on', note=note, velocity=velocity, time=0))
                
            # Note off for all chord notes (after one bar)
            for i, note in enumerate(chord_notes):
                # Last note gets the full duration, others get 0
                off_time = bar_duration if i == len(chord_notes) - 1 else 0
                track.append(Message('note_off', note=note, velocity=0, time=off_time))
            
            time += 1
        
        # Create filename
        filename = f"{raga_name}_lofi_chords.mid"
        midi.save(filename)
        return filename


# Example usage
if __name__ == "__main__":
    generator = RagaMelodyGenerator()
    
    # Generate melodies for different ragas
    for raga in ["yaman", "bhairav", "malkauns"]:
        melody_file = generator.generate_melody(raga, length=32, bpm=75)
        chord_file = generator.generate_chord_progression(raga, length=4, bpm=75)
        print(f"Generated {melody_file} and {chord_file}")