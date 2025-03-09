#!/usr/bin/env python3
"""
Melodic Pattern Generator for Carnatic Ragas
-------------------------------------------
Generates authentic Carnatic melodic patterns based on raga characteristics 
extracted from analysis, suitable for lo-fi music production.
"""

import os
import numpy as np
import json
import pickle
import random
import mido
from mido import Message, MidiFile, MidiTrack
import time
from collections import defaultdict

# Import optional advanced gamaka library if available
try:
    from advanced_gamaka_library import GamakaLibrary
    ADVANCED_GAMAKAS_AVAILABLE = True
except ImportError:
    ADVANCED_GAMAKAS_AVAILABLE = False

class MelodicPatternGenerator:
    """
    Generates melodic patterns based on raga characteristics.
    """
    
    def __init__(self):
        """Initialize the pattern generator."""
        self.raga_models = {}
        self.current_raga_id = None
        self.current_raga_data = None
        
        # Load raga models if available
        self._load_raga_models()
        
        # Initialize gamaka library if available
        self.gamaka_library = None
        if ADVANCED_GAMAKAS_AVAILABLE:
            self.gamaka_library = GamakaLibrary()
    
    def _load_raga_models(self, models_file='data/raga_models.pkl'):
        """Load pre-trained raga models."""
        try:
            with open(models_file, 'rb') as f:
                self.raga_models = pickle.load(f)
            print(f"Loaded {len(self.raga_models)} raga models for pattern generation")
        except (FileNotFoundError, pickle.PickleError):
            print("No raga models found. Loading individual raga feature files...")
            self._load_individual_raga_files()
    
    def _load_individual_raga_files(self, features_dir='data/raga_features'):
        """Load individual raga feature files if available."""
        if not os.path.exists(features_dir):
            print(f"Directory {features_dir} not found. No raga data loaded.")
            return
        
        for filename in os.listdir(features_dir):
            if filename.endswith('.json'):
                try:
                    file_path = os.path.join(features_dir, filename)
                    with open(file_path, 'r') as f:
                        features = json.load(f)
                    
                    # Extract raga ID from metadata or filename
                    if 'metadata' in features and 'raga_id' in features['metadata']:
                        raga_id = features['metadata']['raga_id']
                    else:
                        # Try to extract from filename
                        raga_id = filename.split('_')[0]
                    
                    if raga_id:
                        self.raga_models[raga_id] = features
                    
                except Exception as e:
                    print(f"Error loading raga file {filename}: {e}")
        
        print(f"Loaded {len(self.raga_models)} raga models from individual files")
    
    def load_raga_feature_file(self, feature_file):
        """
        Load raga features from a specific JSON file.
        
        Parameters:
        - feature_file: Path to the JSON file with raga features
        
        Returns:
        - True if successful, False otherwise
        """
        try:
            with open(feature_file, 'r') as f:
                features = json.load(f)
            
            # Extract raga ID from metadata or filename
            if 'metadata' in features and 'raga_id' in features['metadata']:
                raga_id = features['metadata']['raga_id']
            else:
                # Try to extract from filename
                filename = os.path.basename(feature_file)
                raga_id = filename.split('_')[0]
            
            if raga_id:
                self.raga_models[raga_id] = features
                print(f"Loaded raga features for {raga_id}")
                return True
            else:
                print("Could not determine raga ID from the feature file")
                return False
            
        except Exception as e:
            print(f"Error loading raga feature file: {e}")
            return False
    
    def set_current_raga(self, raga_id):
        """
        Set the current raga for pattern generation.
        
        Parameters:
        - raga_id: ID of the raga to use
        
        Returns:
        - True if successful, False otherwise
        """
        if raga_id not in self.raga_models:
            print(f"Raga '{raga_id}' not found in available models")
            return False
        
        self.current_raga_id = raga_id
        self.current_raga_data = self.raga_models[raga_id]
        
        print(f"Set current raga to {raga_id}")
        
        # Print basic info about the raga
        if 'metadata' in self.current_raga_data:
            metadata = self.current_raga_data['metadata']
            print(f"Raga name: {metadata.get('raga_name', 'Unknown')}")
            print(f"Tonic: {metadata.get('tonic', 'Unknown')}")
        
        return True
    
    def list_available_ragas(self):
        """
        List all available ragas in the system.
        
        Returns:
        - List of raga IDs and names
        """
        ragas = []
        
        for raga_id, raga_data in self.raga_models.items():
            name = "Unknown"
            if 'metadata' in raga_data and 'raga_name' in raga_data['metadata']:
                name = raga_data['metadata']['raga_name']
            elif 'name' in raga_data:
                name = raga_data['name']
            
            ragas.append((raga_id, name))
        
        # Sort by name
        ragas.sort(key=lambda x: x[1])
        
        return ragas
    
    def generate_melodic_phrase(self, length=16, seed=None, creativity=0.5):
        """
        Generate a melodic phrase based on the current raga's characteristics.
        
        Parameters:
        - length: Desired length of the phrase in notes
        - seed: Optional seed phrase to start with (list of semitones relative to Sa)
        - creativity: 0.0-1.0 value controlling how closely to follow learned patterns
        
        Returns:
        - List of semitones relative to Sa
        """
        if not self.current_raga_data:
            print("No raga selected. Use set_current_raga() first.")
            return None
        
        # Extract necessary raga features
        arohana_avarohana = self.current_raga_data.get('arohana_avarohana', {})
        transition_matrix = self.current_raga_data.get('transition_matrix', {})
        characteristic_phrases = self.current_raga_data.get('characteristic_phrases', [])
        
        if not arohana_avarohana or not transition_matrix:
            print("Insufficient raga data for pattern generation")
            return None
        
        # 1. Decide approach based on creativity
        if creativity < 0.3 and characteristic_phrases and random.random() < 0.7:
            # Low creativity: Use a characteristic phrase with small variations
            return self._generate_from_characteristic_phrase(length, characteristic_phrases)
        elif creativity < 0.7:
            # Medium creativity: Use transition probabilities but respect raga grammar
            return self._generate_using_transitions(length, seed, transition_matrix, arohana_avarohana)
        else:
            # High creativity: Generate more freely but still within raga scale
            return self._generate_creative_phrase(length, seed, arohana_avarohana)
    
    def _generate_from_characteristic_phrase(self, length, characteristic_phrases):
        """
        Generate a phrase based on characteristic patterns from the raga.
        
        Parameters:
        - length: Desired phrase length
        - characteristic_phrases: List of characteristic phrases for the raga
        
        Returns:
        - Generated phrase as list of semitones
        """
        # Choose a random characteristic phrase
        if not characteristic_phrases:
            return self._generate_creative_phrase(length, None, 
                                                self.current_raga_data.get('arohana_avarohana', {}))
        
        # Weight phrases by count (more frequent ones are more likely)
        weights = [phrase.get('count', 1) for phrase in characteristic_phrases]
        chosen_phrase = random.choices(characteristic_phrases, weights=weights, k=1)[0]
        
        # Get the pattern
        pattern = chosen_phrase['phrase']
        
        # Now extend/trim to desired length
        if len(pattern) >= length:
            # Choose a random starting point and take 'length' notes
            if len(pattern) > length:
                start = random.randint(0, len(pattern) - length)
                result = pattern[start:start + length]
            else:
                result = pattern.copy()
        else:
            # Need to extend - repeat the pattern with small variations
            result = pattern.copy()
            while len(result) < length:
                # 70% chance to repeat with variation, 30% chance to invert direction
                if random.random() < 0.7:
                    # Add a copy with small variations
                    extension = pattern.copy()
                    # Apply variations
                    for i in range(len(extension)):
                        if random.random() < 0.3:
                            # Transpose up or down by 1-2 notes in the scale
                            shift = random.choice([-2, -1, 1, 2])
                            extension[i] = (extension[i] + shift) % 12
                    
                    result.extend(extension)
                else:
                    # Add a reversed version
                    result.extend(list(reversed(pattern)))
        
        # Ensure we have exactly the requested length
        return result[:length]
    
    def _generate_using_transitions(self, length, seed, transition_matrix, arohana_avarohana):
        """
        Generate a phrase using transition probabilities but respecting raga grammar.
        
        Parameters:
        - length: Desired phrase length
        - seed: Optional seed to start with
        - transition_matrix: Matrix of note transition probabilities
        - arohana_avarohana: Ascending/descending patterns
        
        Returns:
        - Generated phrase as list of semitones
        """
        # Initialize with seed or random starting note
        if seed and len(seed) > 0:
            result = seed.copy()
        else:
            # Start with Sa (0) or a common starting note in this raga
            vadi_samvadi = self.current_raga_data.get('vadi_samvadi', {})
            if 'vadi' in vadi_samvadi and vadi_samvadi['vadi'] is not None:
                start_options = [0, vadi_samvadi['vadi']]  # Sa or Vadi
            else:
                start_options = [0]  # Default to Sa
            
            result = [random.choice(start_options)]
        
        # Get available notes in the raga
        arohana = set(arohana_avarohana.get('arohana', [0, 2, 4, 5, 7, 9, 11]))  # Default to major scale
        avarohana = set(arohana_avarohana.get('avarohana', [0, 2, 4, 5, 7, 9, 11]))
        raga_notes = arohana.union(avarohana)
        
        # Get transition matrix data
        matrix = {}
        if 'matrix' in transition_matrix:
            for from_note_str, transitions in transition_matrix['matrix'].items():
                from_note = int(from_note_str)
                matrix[from_note] = {}
                for to_note_str, prob in transitions.items():
                    to_note = int(to_note_str)
                    matrix[from_note][to_note] = prob
        
        # Generate the rest of the phrase
        direction = "ascending"  # Start with ascending
        
        while len(result) < length:
            current_note = result[-1]
            
            # Check if we should change direction
            if direction == "ascending" and current_note >= 7:  # When reaching Pa or higher
                if random.random() < 0.6:  # 60% chance to change to descending
                    direction = "descending"
            elif direction == "descending" and current_note <= 2:  # When reaching Re or lower
                if random.random() < 0.8:  # 80% chance to change to ascending
                    direction = "ascending"
            
            # Get next note options based on transition probabilities
            next_options = []
            next_weights = []
            
            if current_note in matrix:
                for next_note, prob in matrix[current_note].items():
                    next_note = int(next_note)
                    # Only use notes from the correct scale (arohana/avarohana)
                    if (direction == "ascending" and next_note in arohana) or \
                       (direction == "descending" and next_note in avarohana):
                        next_options.append(next_note)
                        next_weights.append(prob)
            
            # If no valid transitions, fall back to scale notes
            if not next_options:
                if direction == "ascending":
                    # Get notes from arohana that are higher than current
                    next_options = [n for n in arohana if (n > current_note and n < current_note + 4) or 
                                   (current_note > 7 and n < 4)]  # Allow wrapping to next octave
                else:
                    # Get notes from avarohana that are lower than current
                    next_options = [n for n in avarohana if (n < current_note and n > current_note - 4) or 
                                   (current_note < 5 and n > 8)]  # Allow wrapping to previous octave
                
                # If still no options, just use any raga note
                if not next_options:
                    next_options = list(raga_notes)
                
                next_weights = [1] * len(next_options)
            
            # Choose next note
            if next_options:
                next_note = random.choices(next_options, weights=next_weights, k=1)[0]
                result.append(next_note)
            else:
                # Fallback to Sa as a last resort
                result.append(0)
        
        return result
    
    def _generate_creative_phrase(self, length, seed, arohana_avarohana):
        """
        Generate a more creative phrase that still respects the raga.
        
        Parameters:
        - length: Desired phrase length
        - seed: Optional seed to start with
        - arohana_avarohana: Ascending/descending patterns
        
        Returns:
        - Generated phrase as list of semitones
        """
        # Initialize with seed or random starting note
        if seed and len(seed) > 0:
            result = seed.copy()
        else:
            # Start with Sa (0) 80% of the time, or another note from the scale
            arohana = arohana_avarohana.get('arohana', [0, 2, 4, 5, 7, 9, 11])
            if random.random() < 0.8:
                result = [0]  # Sa
            else:
                result = [random.choice(arohana)]
        
        # Get raga structure
        arohana = arohana_avarohana.get('arohana', [0, 2, 4, 5, 7, 9, 11])
        avarohana = arohana_avarohana.get('avarohana', [0, 2, 4, 5, 7, 9, 11])
        
        # Create phrase structure - decide phrase shape
        shape = random.choice(["ascending", "descending", "arch", "valley", "zigzag"])
        
        # Generate the phrase according to chosen shape
        if shape == "ascending":
            # Mostly ascending phrase
            self._generate_directional_phrase(result, length, arohana, "up")
        elif shape == "descending":
            # Mostly descending phrase
            self._generate_directional_phrase(result, length, avarohana, "down")
        elif shape == "arch":
            # Ascending then descending
            midpoint = random.randint(length // 3, 2 * length // 3)
            self._generate_directional_phrase(result, midpoint, arohana, "up")
            self._generate_directional_phrase(result, length, avarohana, "down")
        elif shape == "valley":
            # Descending then ascending
            midpoint = random.randint(length // 3, 2 * length // 3)
            self._generate_directional_phrase(result, midpoint, avarohana, "down")
            self._generate_directional_phrase(result, length, arohana, "up")
        else:  # zigzag
            # Alternating short ascending and descending patterns
            while len(result) < length:
                # Short ascending segment
                segment_length = min(random.randint(2, 4), length - len(result))
                if segment_length > 0:
                    self._generate_directional_phrase(result, len(result) + segment_length, arohana, "up")
                
                # Short descending segment
                segment_length = min(random.randint(2, 4), length - len(result))
                if segment_length > 0:
                    self._generate_directional_phrase(result, len(result) + segment_length, avarohana, "down")
        
        # Ensure we have exactly the requested length
        return result[:length]
    
    def _generate_directional_phrase(self, result, target_length, scale, direction):
        """
        Helper method to generate a directional phrase (ascending or descending).
        
        Parameters:
        - result: Current phrase (will be modified)
        - target_length: Target length to reach
        - scale: Scale notes to use
        - direction: "up" or "down"
        """
        while len(result) < target_length:
            current = result[-1]
            
            # Find possible next notes
            if direction == "up":
                # Find notes higher than current
                next_notes = [n for n in scale if n > current]
                if not next_notes:  # Wrap to next octave
                    next_notes = [n for n in scale if n < current]
            else:  # down
                # Find notes lower than current
                next_notes = [n for n in scale if n < current]
                if not next_notes:  # Wrap to previous octave
                    next_notes = [n for n in scale if n > current]
            
            # If still no options, use any note from the scale
            if not next_notes:
                next_notes = scale
            
            # Choose next note - either adjacent or with a leap
            if random.random() < 0.7:  # 70% chance for step-wise motion
                # Find closest note
                closest = min(next_notes, key=lambda x: abs(x - current))
                result.append(closest)
            else:  # 30% chance for a leap
                # Choose a note farther away
                leap_candidates = [n for n in next_notes if abs(n - current) >= 3]
                if leap_candidates:
                    result.append(random.choice(leap_candidates))
                else:
                    # If no leap candidates, use any available note
                    result.append(random.choice(next_notes))
    
    def generate_full_melody(self, length=64, phrases=4, creativity=0.5, gamaka_intensity=0.5):
        """
        Generate a complete melody composed of multiple phrases.
        
        Parameters:
        - length: Total length of the melody in notes
        - phrases: Number of phrases to generate
        - creativity: 0.0-1.0 value controlling how closely to follow learned patterns
        - gamaka_intensity: 0.0-1.0 value controlling the intensity of gamakas
        
        Returns:
        - List of semitones (relative to Sa)
        """
        if not self.current_raga_data:
            print("No raga selected. Use set_current_raga() first.")
            return None
        
        # Calculate phrase length
        phrase_length = length // phrases
        
        # Generate first phrase
        melody = self.generate_melodic_phrase(phrase_length, None, creativity)
        
        # Generate subsequent phrases
        for i in range(1, phrases):
            # Use the end of the previous phrase as seed for the next
            seed_length = min(3, len(melody))
            seed = melody[-seed_length:]
            
            # Adjust creativity to occasionally go beyond patterns
            adjusted_creativity = min(1.0, creativity + random.uniform(-0.2, 0.2))
            
            # Generate next phrase
            next_phrase = self.generate_melodic_phrase(phrase_length, seed, adjusted_creativity)
            
            # Ensure smooth connection between phrases
            if next_phrase and melody:
                # Sometimes add a connecting note
                if random.random() < 0.3:
                    connector = self._get_connecting_note(melody[-1], next_phrase[0])
                    melody.append(connector)
            
            # Add the new phrase
            if next_phrase:
                melody.extend(next_phrase)
        
        # Apply gamakas if the library is available
        if self.gamaka_library and gamaka_intensity > 0:
            melody_with_gamakas = self._apply_gamakas(melody, gamaka_intensity)
            return melody_with_gamakas
        
        return melody[:length]  # Ensure exact length
    
    def _get_connecting_note(self, last_note, next_note):
        """
        Get a suitable connecting note between two phrases.
        
        Parameters:
        - last_note: Last note of the previous phrase
        - next_note: First note of the next phrase
        
        Returns:
        - A connecting note
        """
        # Get raga notes
        arohana_avarohana = self.current_raga_data.get('arohana_avarohana', {})
        arohana = set(arohana_avarohana.get('arohana', [0, 2, 4, 5, 7, 9, 11]))
        avarohana = set(arohana_avarohana.get('avarohana', [0, 2, 4, 5, 7, 9, 11]))
        all_notes = arohana.union(avarohana)
        
        # If the interval is small, use a note in between if available
        interval = (next_note - last_note) % 12
        
        if interval == 0:
            # Same note - use another note from the scale
            candidates = [n for n in all_notes if n != last_note]
            if candidates:
                return random.choice(candidates)
            else:
                return (last_note + 7) % 12  # Perfect fifth
        elif interval <= 2:
            # Small interval - can keep as is
            return next_note
        elif interval >= 10:
            # Descending step - can keep as is
            return next_note
        else:
            # Larger interval - find a connecting note
            if next_note > last_note:
                # Ascending
                midpoint = (last_note + next_note) // 2
                candidates = [n for n in all_notes if last_note < n < next_note]
            else:
                # Descending (with wrap around octave)
                midpoint = (last_note + next_note + 12) // 2 % 12
                candidates = [n for n in all_notes if n > last_note or n < next_note]
            
            if candidates:
                return random.choice(candidates)
            else:
                return midpoint % 12
    
    def _apply_gamakas(self, melody, intensity):
        """
        Apply characteristic gamakas to notes based on raga.
        
        Parameters:
        - melody: The melody to ornament
        - intensity: 0.0-1.0 value for gamaka intensity
        
        Returns:
        - List of notes with gamaka information attached
        """
        if not self.gamaka_library:
            return melody
        
        # Get the result from the gamaka library
        result = self.gamaka_library.apply_gamaka_sequence(melody, self.current_raga_id, intensity)
        
        # Extract just the notes (without bend information)
        ornamented_melody = [event['note'] for event in result]
        
        return ornamented_melody
    
    def create_midi_sequence(self, melody, filename=None, base_note=60, bpm=75, apply_gamakas=True):
        """
        Create a MIDI file from the generated melody.
        
        Parameters:
        - melody: The melody to convert (list of semitones)
        - filename: Output filename (or None for auto-generated)
        - base_note: MIDI note number for Sa (usually 60 = C4)
        - bpm: Tempo in beats per minute
        - apply_gamakas: Whether to apply gamakas to the MIDI
        
        Returns:
        - Filename of the generated MIDI file
        """
        if not melody:
            print("No melody provided")
            return None
        
        # Create MIDI file
        midi = MidiFile()
        track = MidiTrack()
        midi.tracks.append(track)
        
        # Add tempo
        tempo = mido.bpm2tempo(bpm)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))
        
        # Add track name
        if self.current_raga_id:
            raga_name = "Unknown"
            if 'metadata' in self.current_raga_data and 'raga_name' in self.current_raga_data['metadata']:
                raga_name = self.current_raga_data['metadata']['raga_name']
            track_name = f"{raga_name} ({self.current_raga_id}) Melody"
        else:
            track_name = "Generated Melody"
        
        track.append(mido.MetaMessage('track_name', name=track_name))
        
        # Add time signature (4/4 for lo-fi)
        track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))
        
        # Generate notes with gamakas if both required and available
        use_gamakas = apply_gamakas and self.gamaka_library and ADVANCED_GAMAKAS_AVAILABLE
        
        if use_gamakas:
            # Get melody with detailed gamaka information
            gamaka_result = self.gamaka_library.apply_gamaka_sequence(
                melody, self.current_raga_id, intensity=0.7
            )
            self._add_notes_with_gamakas(track, gamaka_result, base_note)
        else:
            # Add regular notes without gamakas
            self._add_simple_notes(track, melody, base_note)
        
        # Create filename
        if filename is None:
            timestamp = int(time.time())
            if self.current_raga_id:
                filename = f"outputs/{self.current_raga_id}_melody_{timestamp}.mid"
            else:
                filename = f"outputs/generated_melody_{timestamp}.mid"
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save MIDI file
        midi.save(filename)
        print(f"MIDI sequence saved to {filename}")
        
        return filename
    
    def _add_simple_notes(self, track, melody, base_note):
        """
        Add notes to the MIDI track without complex gamakas.
        
        Parameters:
        - track: MIDI track to add notes to
        - melody: List of semitones
        - base_note: Base MIDI note number for Sa
        """
        # Add notes with some velocity variation for human feel
        ticks_per_beat = 480  # Standard MIDI resolution
        duration = ticks_per_beat // 2  # Eighth notes
        
        for note_value in melody:
            # Calculate MIDI note number (Sa = base_note)
            midi_note = base_note + note_value
            
            # Add some velocity variation
            velocity = random.randint(70, 95)
            
            # Add some duration variation
            note_duration = duration + random.randint(-30, 30)
            
            # Note on
            track.append(Message('note_on', note=midi_note, velocity=velocity, time=0))
            
            # Note off
            track.append(Message('note_off', note=midi_note, velocity=0, time=max(1, note_duration)))
    
    def _add_notes_with_gamakas(self, track, gamaka_result, base_note):
        """
        Add notes to the MIDI track with complex gamakas.
        
        Parameters:
        - track: MIDI track to add notes to
        - gamaka_result: List of note events with gamaka information
        - base_note: Base MIDI note number for Sa
        """
        # Advanced note addition with pitch bends
        ticks_per_beat = 480  # Standard MIDI resolution
        duration = ticks_per_beat // 2  # Eighth notes
        
        # MIDI uses a pitch bend range of -8192 to +8191
        # We'll scale our semitone-based bends to this range
        # Standard pitch bend range is ±2 semitones (but this can vary by synth)
        BEND_RANGE = 2  # semitones
        MAX_BEND = 8191
        
        for note_event in gamaka_result:
            note_value = note_event['note']
            bend_data = note_event['bend']
            
            # Calculate MIDI note number
            midi_note = base_note + note_value
            
            # Humanize velocity and timing
            velocity = random.randint(70, 95)
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
            note_duration = duration + random.randint(-30, 30)
            track.append(Message('note_off', note=midi_note, velocity=0, 
                        time=max(1, note_duration)))
    
    def generate_accompaniment(self, melody, filename=None, base_note=48, bpm=75):
        """
        Generate a simple chord accompaniment based on the melody.
        
        Parameters:
        - melody: The melody (list of semitones)
        - filename: Output filename (or None for auto-generated)
        - base_note: MIDI note number for Sa for the chords (usually lower than melody)
        - bpm: Tempo in beats per minute
        
        Returns:
        - Filename of the generated MIDI file
        """
        if not melody:
            print("No melody provided")
            return None
        
        # Get raga characteristics
        if not self.current_raga_data:
            print("No raga selected. Using generic accompaniment.")
            arohana = [0, 2, 4, 5, 7, 9, 11]  # Default to major scale
        else:
            arohana_avarohana = self.current_raga_data.get('arohana_avarohana', {})
            arohana = arohana_avarohana.get('arohana', [0, 2, 4, 5, 7, 9, 11])
        
        # Create MIDI file
        midi = MidiFile()
        track = MidiTrack()
        midi.tracks.append(track)
        
        # Add tempo
        tempo = mido.bpm2tempo(bpm)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))
        
        # Add track name
        if self.current_raga_id:
            track_name = f"{self.current_raga_id} Accompaniment"
        else:
            track_name = "Generated Accompaniment"
        
        track.append(mido.MetaMessage('track_name', name=track_name))
        
        # Add time signature (4/4 for lo-fi)
        track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))
        
        # Divide melody into segments for different chords
        segment_length = 8  # 8 notes per chord
        
        # Generate chords for each segment
        for i in range(0, len(melody), segment_length):
            segment = melody[i:i+segment_length]
            if not segment:
                continue
                
            # Find most common notes in this segment
            note_counts = Counter(segment)
            common_notes = [note for note, count in note_counts.most_common(3)]
            
            # Generate chord based on common notes
            chord = self._generate_chord_for_segment(common_notes, arohana)
            
            # Add chord to track
            self._add_chord_to_track(track, chord, base_note)
        
        # Create filename
        if filename is None:
            timestamp = int(time.time())
            if self.current_raga_id:
                filename = f"outputs/{self.current_raga_id}_accompaniment_{timestamp}.mid"
            else:
                filename = f"outputs/generated_accompaniment_{timestamp}.mid"
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save MIDI file
        midi.save(filename)
        print(f"Accompaniment saved to {filename}")
        
        return filename
    
    def _generate_chord_for_segment(self, common_notes, scale):
        """
        Generate a chord that complements the given notes.
        
        Parameters:
        - common_notes: Most common notes in the segment
        - scale: Scale notes to use
        
        Returns:
        - List of semitones for the chord
        """
        # Ensure we have at least one note
        if not common_notes:
            return [0, 4, 7]  # Default to Sa-Ma-Pa
        
        # Start with the most common note as root
        root = common_notes[0]
        chord = [root]
        
        # Find suitable thirds and fifths from the scale
        for interval in [3, 4, 7]:  # Try both minor and major thirds, and fifth
            note = (root + interval) % 12
            if note in scale:
                chord.append(note)
        
        # If chord has less than 3 notes, add other notes from the scale
        while len(chord) < 3:
            # Try to add another note from scale that creates a consonant interval
            candidates = [note for note in scale if note not in chord]
            
            if not candidates:
                break
                
            # Add note that forms consonant interval with root (4, 5, 7 semitones)
            consonant_candidates = [note for note in candidates if 
                                   (note - root) % 12 in [3, 4, 5, 7, 8, 9]]
            
            if consonant_candidates:
                chord.append(random.choice(consonant_candidates))
            else:
                chord.append(random.choice(candidates))
        
        return chord
    
    def _add_chord_to_track(self, track, chord, base_note):
        """
        Add a chord to the MIDI track.
        
        Parameters:
        - track: MIDI track to add chord to
        - chord: List of semitones for the chord
        - base_note: Base MIDI note number for Sa
        """
        # Standard MIDI resolution
        ticks_per_beat = 480
        
        # Chord duration (4 beats = 1 bar)
        duration = ticks_per_beat * 4
        
        # For lo-fi feel, use soft velocity
        velocity = random.randint(40, 60)
        
        # Add all notes in the chord simultaneously
        for note_value in chord:
            midi_note = base_note + note_value
            
            # Note on
            track.append(Message('note_on', note=midi_note, velocity=velocity, time=0))
        
        # Add note-offs with proper timing
        # First notes have 0 time (simultaneous), last note has full duration
        for i, note_value in enumerate(chord):
            midi_note = base_note + note_value
            
            # Note off - last note gets the full duration, others get 0
            time_value = duration if i == len(chord) - 1 else 0
            track.append(Message('note_off', note=midi_note, velocity=0, time=time_value))
    
    def generate_complete_track(self, length=64, base_note=60, bpm=75, creativity=0.5, 
                               gamaka_intensity=0.5):
        """
        Generate a complete track with melody and accompaniment.
        
        Parameters:
        - length: Total length of the melody in notes
        - base_note: Base MIDI note number for Sa
        - bpm: Tempo in beats per minute
        - creativity: 0.0-1.0 value controlling how closely to follow learned patterns
        - gamaka_intensity: 0.0-1.0 value controlling the intensity of gamakas
        
        Returns:
        - Dictionary with filenames for each component
        """
        if not self.current_raga_id:
            print("No raga selected. Use set_current_raga() first.")
            return None
        
        # Generate melody
        melody = self.generate_full_melody(length, phrases=4, creativity=creativity, 
                                          gamaka_intensity=gamaka_intensity)
        
        # Create timestamp for consistent filenames
        timestamp = int(time.time())
        
        # Create MIDI files
        melody_file = self.create_midi_sequence(
            melody, 
            filename=f"outputs/{self.current_raga_id}_melody_{timestamp}.mid",
            base_note=base_note,
            bpm=bpm
        )
        
        # Generate accompaniment
        accompaniment_file = self.generate_accompaniment(
            melody,
            filename=f"outputs/{self.current_raga_id}_accompaniment_{timestamp}.mid",
            base_note=base_note-12,  # One octave lower
            bpm=bpm
        )
        
        # Return file information
        raga_name = "Unknown"
        if 'metadata' in self.current_raga_data and 'raga_name' in self.current_raga_data['metadata']:
            raga_name = self.current_raga_data['metadata']['raga_name']
        
        return {
            'melody': melody_file,
            'accompaniment': accompaniment_file,
            'raga': raga_name,
            'raga_id': self.current_raga_id,
            'bpm': bpm,
            'timestamp': timestamp
        }


def generate_raga_melody(raga_id, output_dir='outputs', length=64, base_note=60, bpm=75,
                       creativity=0.5, gamaka_intensity=0.5):
    """
    Utility function to generate a melody based on a specific raga.
    
    Parameters:
    - raga_id: ID of the raga to use
    - output_dir: Directory for output files
    - length: Length of the melody to generate
    - base_note: Base MIDI note number for Sa
    - bpm: Tempo in beats per minute
    - creativity: 0.0-1.0 value controlling how closely to follow learned patterns
    - gamaka_intensity: 0.0-1.0 value controlling the intensity of gamakas
    
    Returns:
    - Dictionary with generated file information
    """
    # Create generator
    generator = MelodicPatternGenerator()
    
    # Set raga
    if not generator.set_current_raga(raga_id):
        print(f"Raga '{raga_id}' not found. Available ragas:")
        for raga_id, name in generator.list_available_ragas():
            print(f"  {raga_id}: {name}")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate complete track
    result = generator.generate_complete_track(
        length=length,
        base_note=base_note,
        bpm=bpm,
        creativity=creativity,
        gamaka_intensity=gamaka_intensity
    )
    
    return result


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        raga_id = sys.argv[1]
        
        # Parse optional parameters
        length = 64
        creativity = 0.5
        gamaka_intensity = 0.5
        
        if len(sys.argv) > 2:
            try:
                length = int(sys.argv[2])
            except ValueError:
                print(f"Invalid length value: {sys.argv[2]}. Using default: 64")
        
        if len(sys.argv) > 3:
            try:
                creativity = float(sys.argv[3])
            except ValueError:
                print(f"Invalid creativity value: {sys.argv[3]}. Using default: 0.5")
        
        if len(sys.argv) > 4:
            try:
                gamaka_intensity = float(sys.argv[4])
            except ValueError:
                print(f"Invalid gamaka intensity value: {sys.argv[4]}. Using default: 0.5")
        
        # Generate melody
        result = generate_raga_melody(
            raga_id, 
            length=length, 
            creativity=creativity,
            gamaka_intensity=gamaka_intensity
        )
        
        if result:
            print("\nGeneration Results:")
            print(f"Raga: {result['raga']} ({result['raga_id']})")
            print(f"Melody file: {result['melody']}")
            print(f"Accompaniment file: {result['accompaniment']}")
    else:
        # Display available ragas if no argument provided
        generator = MelodicPatternGenerator()
        available_ragas = generator.list_available_ragas()
        
        if available_ragas:
            print("Available ragas for melody generation:")
            for raga_id, name in available_ragas:
                print(f"  {raga_id}: {name}")
            print("\nUsage: python melodic_pattern_generator.py <raga_id> [length] [creativity] [gamaka_intensity]")
        else:
            print("No raga models found. Please analyze some recordings first.")
            print("Usage: python melodic_pattern_generator.py <raga_id> [length] [creativity] [gamaka_intensity]")