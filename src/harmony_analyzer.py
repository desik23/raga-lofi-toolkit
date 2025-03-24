#!/usr/bin/env python3
"""
Harmony Analyzer for Raga-Lofi Integration
-----------------------------------------
Analyzes and generates harmony structures compatible with raga-based melodies,
bridging Western harmony concepts with Indian classical music for lofi production.
"""

import os
import numpy as np
import librosa
import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import mido

class HarmonyAnalyzer:
    """
    Analyzes harmonic content in audio and MIDI files, with specialized
    capabilities for integrating Western harmony with raga-based music.
    """
    
    def __init__(self, raga_data_path='data/ragas.json'):
        """
        Initialize the harmony analyzer.
        
        Parameters:
        - raga_data_path: Path to raga definition file
        """
        self.chord_profiles = self._initialize_chord_profiles()
        self.raga_harmony_mappings = {}
        self.analyzed_harmony = None
        
        # Load raga data
        self._load_raga_data(raga_data_path)
    
    def _load_raga_data(self, raga_data_path):
        """Load raga definitions and create harmonic mappings."""
        try:
            with open(raga_data_path, 'r') as f:
                data = json.load(f)
                self.ragas = {raga['id']: raga for raga in data['ragas']}
            
            # Generate harmonic mappings for each raga
            for raga_id, raga in self.ragas.items():
                self.raga_harmony_mappings[raga_id] = self._create_raga_harmony_mapping(raga)
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load raga data: {e}")
            self.ragas = {}
    
    def _initialize_chord_profiles(self):
        """
        Initialize spectral profiles for common chord types.
        Returns a dictionary of chord profiles.
        """
        # Chroma profiles for common chord types
        profiles = {
            # Major chords (1-3-5)
            'major': np.zeros(12),
            
            # Minor chords (1-b3-5)
            'minor': np.zeros(12),
            
            # Dominant 7th (1-3-5-b7)
            'dom7': np.zeros(12),
            
            # Minor 7th (1-b3-5-b7)
            'min7': np.zeros(12),
            
            # Major 7th (1-3-5-7)
            'maj7': np.zeros(12),
            
            # Suspended 4th (1-4-5)
            'sus4': np.zeros(12),
            
            # Suspended 2nd (1-2-5)
            'sus2': np.zeros(12),
            
            # Diminished (1-b3-b5)
            'dim': np.zeros(12),
            
            # Half-diminished (1-b3-b5-b7)
            'half_dim': np.zeros(12),
            
            # Augmented (1-3-#5)
            'aug': np.zeros(12)
        }
        
        # Set values for major chord (relative to C)
        profiles['major'][[0, 4, 7]] = [1.0, 0.8, 0.9]
        
        # Set values for minor chord
        profiles['minor'][[0, 3, 7]] = [1.0, 0.8, 0.9]
        
        # Set values for dominant 7th
        profiles['dom7'][[0, 4, 7, 10]] = [1.0, 0.8, 0.9, 0.7]
        
        # Set values for minor 7th
        profiles['min7'][[0, 3, 7, 10]] = [1.0, 0.8, 0.9, 0.7]
        
        # Set values for major 7th
        profiles['maj7'][[0, 4, 7, 11]] = [1.0, 0.8, 0.9, 0.7]
        
        # Set values for suspended 4th
        profiles['sus4'][[0, 5, 7]] = [1.0, 0.8, 0.9]
        
        # Set values for suspended 2nd
        profiles['sus2'][[0, 2, 7]] = [1.0, 0.8, 0.9]
        
        # Set values for diminished
        profiles['dim'][[0, 3, 6]] = [1.0, 0.8, 0.9]
        
        # Set values for half-diminished
        profiles['half_dim'][[0, 3, 6, 10]] = [1.0, 0.8, 0.9, 0.7]
        
        # Set values for augmented
        profiles['aug'][[0, 4, 8]] = [1.0, 0.8, 0.9]
        
        return profiles
    
    def _create_raga_harmony_mapping(self, raga):
        """
        Create a mapping of Western harmony concepts for a specific raga.
        
        Parameters:
        - raga: Raga definition dictionary
        
        Returns:
        - Dictionary with harmony mapping information
        """
        if not raga:
            return {}
            
        # Extract arohana and avarohana (ascending/descending scales)
        arohan = raga.get('arohan', [])
        avarohan = raga.get('avarohan', [])
        
        # Get unique notes in the raga
        raga_notes = sorted(set([n % 12 for n in arohan + avarohan]))
        
        # Identify compatible chord types
        compatible_chords = self._find_compatible_chords(raga_notes)
        
        # Create progressions based on raga characteristics
        progressions = self._create_raga_progressions(raga_notes, raga)
        
        # Determine dissonance/consonance relationships specific to this raga
        consonance_map = self._create_consonance_map(raga_notes, raga)
        
        return {
            'raga_notes': raga_notes,
            'compatible_chords': compatible_chords,
            'suggested_progressions': progressions,
            'consonance_map': consonance_map,
            'vadi': raga.get('vadi'),
            'samvadi': raga.get('samvadi')
        }
    
    def _find_compatible_chords(self, raga_notes):
        """
        Find chord types compatible with the given raga notes.
        
        Parameters:
        - raga_notes: List of raga notes as scale degrees (0-11)
        
        Returns:
        - Dictionary of compatible chord types by root note
        """
        compatible = {}
        
        # Check each possible root note
        for root in range(12):
            compatible[root] = []
            
            # Check major chord
            if all((root + offset) % 12 in raga_notes for offset in [0, 4, 7]):
                compatible[root].append('major')
                
            # Check minor chord
            if all((root + offset) % 12 in raga_notes for offset in [0, 3, 7]):
                compatible[root].append('minor')
                
            # Check dominant 7th
            if all((root + offset) % 12 in raga_notes for offset in [0, 4, 7, 10]):
                compatible[root].append('dom7')
                
            # Check minor 7th
            if all((root + offset) % 12 in raga_notes for offset in [0, 3, 7, 10]):
                compatible[root].append('min7')
                
            # Check major 7th
            if all((root + offset) % 12 in raga_notes for offset in [0, 4, 7, 11]):
                compatible[root].append('maj7')
                
            # Check suspended 4th
            if all((root + offset) % 12 in raga_notes for offset in [0, 5, 7]):
                compatible[root].append('sus4')
                
            # Check suspended 2nd
            if all((root + offset) % 12 in raga_notes for offset in [0, 2, 7]):
                compatible[root].append('sus2')
        
        return compatible
    
    def _create_raga_progressions(self, raga_notes, raga):
        """
        Create chord progressions suitable for the raga.
        
        Parameters:
        - raga_notes: List of raga notes as scale degrees (0-11)
        - raga: Raga definition dictionary
        
        Returns:
        - List of suggested chord progressions
        """
        progressions = []
        
        # Get important notes (vadi, samvadi, graha)
        vadi = raga.get('vadi')
        samvadi = raga.get('samvadi')
        
        # If we have vadi and samvadi, use them as primary harmony points
        if vadi is not None and samvadi is not None:
            # Simple I-IV-V type progression adapted to raga
            if vadi in raga_notes and samvadi in raga_notes:
                # Create progression centered on vadi
                prog = [
                    {"root": vadi, "type": "major" if (vadi + 4) % 12 in raga_notes else "minor"},
                    {"root": samvadi, "type": "major" if (samvadi + 4) % 12 in raga_notes else "minor"},
                    {"root": vadi, "type": "major" if (vadi + 4) % 12 in raga_notes else "minor"}
                ]
                progressions.append(prog)
        
        # Add some generic lo-fi friendly progressions based on available notes
        tonic = 0  # Sa
        if tonic in raga_notes:
            fifth = 7  # Pa
            fourth = 5  # Ma
            
            # Common lo-fi progression adapted to raga
            if fifth in raga_notes and fourth in raga_notes:
                prog = [
                    {"root": tonic, "type": "major" if (tonic + 4) % 12 in raga_notes else "minor"},
                    {"root": fourth, "type": "major" if (fourth + 4) % 12 in raga_notes else "minor"},
                    {"root": fifth, "type": "major" if (fifth + 4) % 12 in raga_notes else "minor"},
                    {"root": tonic, "type": "major" if (tonic + 4) % 12 in raga_notes else "minor"}
                ]
                progressions.append(prog)
        
        # Add ambient/modal progressions for texture
        if len(raga_notes) >= 4:
            # Find the most consonant intervals in the raga
            consonant_intervals = [7, 5, 4]  # Perfect 5th, Perfect 4th, Major 3rd
            consonant_roots = []
            
            for interval in consonant_intervals:
                if interval in raga_notes:
                    consonant_roots.append(interval)
            
            if consonant_roots:
                # Create ambient progression with sustained chords
                prog = [
                    {"root": tonic, "type": "sus2" if 2 in raga_notes else "sus4" if 5 in raga_notes else "major"},
                    {"root": consonant_roots[0], "type": "sus2" if (consonant_roots[0] + 2) % 12 in raga_notes else "sus4"}
                ]
                progressions.append(prog)
        
        return progressions
    
    def _create_consonance_map(self, raga_notes, raga):
        """
        Create a map of consonance relationships between notes in the raga.
        
        Parameters:
        - raga_notes: List of raga notes as scale degrees (0-11)
        - raga: Raga definition dictionary
        
        Returns:
        - Dictionary mapping each note to its consonance relationships
        """
        consonance_map = {}
        
        # Define consonant intervals in Western music (could be extended with microtonal adjustments)
        perfect_consonance = [0, 7]  # Unison/octave, perfect fifth
        imperfect_consonance = [3, 4, 8, 9]  # Minor/major thirds, minor/major sixths
        
        for note in raga_notes:
            consonance_map[note] = {
                "perfect": [
                    (note + interval) % 12 for interval in perfect_consonance
                    if (note + interval) % 12 in raga_notes
                ],
                "imperfect": [
                    (note + interval) % 12 for interval in imperfect_consonance
                    if (note + interval) % 12 in raga_notes
                ],
                "dissonant": [
                    other_note for other_note in raga_notes
                    if (other_note - note) % 12 not in perfect_consonance + imperfect_consonance
                ]
            }
        
        return consonance_map
    
    def analyze_audio(self, file_path, tonic_freq=None):
        """
        Analyze harmonic content in an audio file.
        
        Parameters:
        - file_path: Path to the audio file
        - tonic_freq: Tonic frequency in Hz (if known)
        
        Returns:
        - Dictionary with harmonic analysis results
        """
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: Audio file not found: {file_path}")
            return None
        
        try:
            # Load the audio file
            y, sr = librosa.load(file_path, sr=22050, mono=True)
            
            # Compute chromagram
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            
            # If tonic not provided, try to estimate it
            if tonic_freq is None:
                tonic_freq = self._estimate_tonic(y, sr)
            
            # Convert tonic frequency to chroma bin
            if tonic_freq:
                tonic_chroma = self._freq_to_chroma_bin(tonic_freq)
            else:
                tonic_chroma = None
            
            # Extract harmony segments using librosa
            self.analyzed_harmony = self._extract_harmony_segments(y, sr, chroma, tonic_chroma)
            
            return self.analyzed_harmony
            
        except Exception as e:
            print(f"Error analyzing audio: {e}")
            return None
    
    def analyze_midi(self, midi_path):
        """
        Analyze harmonic content in a MIDI file.
        
        Parameters:
        - midi_path: Path to the MIDI file
        
        Returns:
        - Dictionary with harmonic analysis results
        """
        if not os.path.exists(midi_path):
            print(f"Error: MIDI file not found: {midi_path}")
            return None
        
        try:
            # Load the MIDI file
            midi_data = mido.MidiFile(midi_path)
            
            # Extract note events
            notes = []
            for track in midi_data.tracks:
                current_time = 0
                for msg in track:
                    current_time += msg.time
                    if msg.type == 'note_on' and msg.velocity > 0:
                        # Store note, start time, and track
                        notes.append({
                            'note': msg.note % 12,  # Convert to pitch class
                            'time': current_time,
                            'track': track.name if hasattr(track, 'name') else "Unknown"
                        })
            
            # Group notes into temporal segments
            segments = self._segment_midi_notes(notes, midi_data.ticks_per_beat)
            
            # Analyze each segment for harmony
            harmony_segments = []
            for segment in segments:
                notes_in_segment = [note['note'] for note in segment['notes']]
                chord = self._identify_chord_from_notes(notes_in_segment)
                harmony_segments.append({
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'notes': notes_in_segment,
                    'chord': chord
                })
            
            self.analyzed_harmony = {
                'file': midi_path,
                'harmony_segments': harmony_segments,
                'predominant_chords': self._find_predominant_chords(harmony_segments),
                'chord_progression': self._extract_chord_progression(harmony_segments)
            }
            
            return self.analyzed_harmony
            
        except Exception as e:
            print(f"Error analyzing MIDI: {e}")
            return None
    
    def _estimate_tonic(self, y, sr):
        """
        Estimate the tonic frequency from an audio signal.
        
        Parameters:
        - y: Audio signal
        - sr: Sample rate
        
        Returns:
        - Estimated tonic frequency
        """
        # Using PYIN algorithm for pitch estimation
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), 
                                                   fmax=librosa.note_to_hz('C6'), 
                                                   sr=sr)
        
        # Filter out unvoiced frames
        f0_valid = f0[voiced_flag]
        
        if len(f0_valid) == 0:
            return None
        
        # Convert to pitch class distribution
        cents = 1200 * np.log2(f0_valid / librosa.note_to_hz('C1'))
        cents_wrapped = cents % 1200  # Wrap to one octave
        
        # Create histogram of pitch classes
        hist, _ = np.histogram(cents_wrapped, bins=np.linspace(0, 1200, 121))  # 10 cent resolution
        
        # Smooth histogram
        hist_smooth = np.convolve(hist, np.hanning(5)/np.sum(np.hanning(5)), mode='same')
        
        # Find the most common pitch class
        max_idx = np.argmax(hist_smooth)
        tonic_cents = max_idx * 10  # Convert bin index to cents
        
        # Convert back to frequency
        tonic_freq = librosa.note_to_hz('C1') * 2**(tonic_cents/1200)
        
        return tonic_freq
    
    def _freq_to_chroma_bin(self, freq):
        """Convert frequency to chroma bin."""
        note = librosa.hz_to_note(freq)
        # Extract the note class (without octave)
        note_class = note[:-1]
        
        # Map to chroma bin
        note_map = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 
                    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
        
        return note_map.get(note_class, 0)
    
    def _extract_harmony_segments(self, y, sr, chroma, tonic_chroma=None):
        """
        Extract harmony segments from audio using chroma features.
        
        Parameters:
        - y: Audio signal
        - sr: Sample rate
        - chroma: Chroma features
        - tonic_chroma: Tonic note chroma bin (if known)
        
        Returns:
        - Dictionary with harmony analysis results
        """
        # Calculate the average chroma vector across the entire piece
        avg_chroma = np.mean(chroma, axis=1)
        
        # Normalize for comparison
        avg_chroma = avg_chroma / np.sum(avg_chroma)
        
        # Segment the chroma into temporal sections
        frame_length = 8  # Number of frames to combine
        num_segments = chroma.shape[1] // frame_length
        
        harmony_segments = []
        
        for i in range(num_segments):
            start_frame = i * frame_length
            end_frame = (i + 1) * frame_length
            
            # Calculate segment chroma
            segment_chroma = np.mean(chroma[:, start_frame:end_frame], axis=1)
            segment_chroma = segment_chroma / np.sum(segment_chroma)
            
            # Identify chord for this segment
            chord = self._identify_chord(segment_chroma, tonic_chroma)
            
            # Store segment data
            harmony_segments.append({
                'start_time': librosa.frames_to_time(start_frame, sr=sr),
                'end_time': librosa.frames_to_time(end_frame, sr=sr),
                'chroma': segment_chroma.tolist(),
                'chord': chord
            })
        
        # Find predominant chords
        predominant_chords = self._find_predominant_chords(harmony_segments)
        
        # Extract chord progression
        chord_progression = self._extract_chord_progression(harmony_segments)
        
        return {
            'tonic_chroma': tonic_chroma,
            'avg_chroma': avg_chroma.tolist(),
            'harmony_segments': harmony_segments,
            'predominant_chords': predominant_chords,
            'chord_progression': chord_progression
        }
    
    def _identify_chord(self, chroma, tonic_chroma=None):
        """
        Identify the chord from a chroma vector.
        
        Parameters:
        - chroma: Chroma vector
        - tonic_chroma: Tonic note chroma bin (if known)
        
        Returns:
        - Dictionary with chord information
        """
        # Find the most prominent notes
        threshold = 0.1
        prominent_notes = [i for i, v in enumerate(chroma) if v > threshold]
        
        # Try to identify the root note (highest energy or first note)
        if len(prominent_notes) > 0:
            root_candidates = sorted([(i, chroma[i]) for i in prominent_notes], 
                                   key=lambda x: x[1], reverse=True)
            root = root_candidates[0][0]
        else:
            # Default to tonic if available, otherwise C
            root = tonic_chroma if tonic_chroma is not None else 0
        
        # Match against chord profiles to identify chord type
        best_match = None
        best_score = -1
        
        for chord_type, profile in self.chord_profiles.items():
            # Rotate profile to match the root
            rotated_profile = np.roll(profile, root)
            
            # Calculate similarity score (dot product)
            score = np.dot(chroma, rotated_profile)
            
            if score > best_score:
                best_score = score
                best_match = chord_type
        
        # Convert root to note name
        root_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][root]
        
        return {
            'root': root,
            'root_name': root_name,
            'type': best_match,
            'notes': prominent_notes,
            'confidence': best_score
        }
    
    def _segment_midi_notes(self, notes, ticks_per_beat, min_notes_per_segment=3):
        """
        Group MIDI notes into temporal segments for harmonic analysis.
        
        Parameters:
        - notes: List of note events
        - ticks_per_beat: MIDI ticks per beat
        - min_notes_per_segment: Minimum notes required to form a segment
        
        Returns:
        - List of segments with notes
        """
        if not notes:
            return []
        
        # Sort notes by time
        notes.sort(key=lambda x: x['time'])
        
        # Group notes into temporal clusters
        segments = []
        current_segment = {'notes': [], 'start_time': notes[0]['time']}
        
        # Threshold for new segment (one beat)
        time_threshold = ticks_per_beat
        
        for i, note in enumerate(notes):
            # If this note is far from the previous one, start a new segment
            if i > 0 and note['time'] - notes[i-1]['time'] > time_threshold:
                if len(current_segment['notes']) >= min_notes_per_segment:
                    current_segment['end_time'] = notes[i-1]['time']
                    segments.append(current_segment)
                
                current_segment = {'notes': [], 'start_time': note['time']}
            
            current_segment['notes'].append(note)
        
        # Add the last segment
        if len(current_segment['notes']) >= min_notes_per_segment:
            current_segment['end_time'] = notes[-1]['time']
            segments.append(current_segment)
        
        return segments
    
    def _identify_chord_from_notes(self, notes):
        """
        Identify chord from a set of MIDI note numbers.
        
        Parameters:
        - notes: List of note pitch classes (0-11)
        
        Returns:
        - Dictionary with chord information
        """
        # Count occurrences of each pitch class
        note_counts = Counter(notes)
        
        # Create a chroma-like vector
        chroma = np.zeros(12)
        for note, count in note_counts.items():
            chroma[note] = count
        
        # Normalize
        if np.sum(chroma) > 0:
            chroma = chroma / np.sum(chroma)
        
        # Identify chord using the chroma vector
        return self._identify_chord(chroma)
    
    def _find_predominant_chords(self, harmony_segments):
        """
        Find the most commonly occurring chords in the harmony segments.
        
        Parameters:
        - harmony_segments: List of harmony segments
        
        Returns:
        - List of predominant chords with counts
        """
        chord_counts = Counter()
        
        for segment in harmony_segments:
            chord = segment['chord']
            chord_key = f"{chord['root_name']} {chord['type']}"
            chord_counts[chord_key] += 1
        
        # Return most common chords
        return chord_counts.most_common(5)
    
    def _extract_chord_progression(self, harmony_segments):
        """
        Extract the main chord progression from harmony segments.
        
        Parameters:
        - harmony_segments: List of harmony segments
        
        Returns:
        - List of chords representing the main progression
        """
        if not harmony_segments:
            return []
        
        # Simplify consecutive identical chords
        progression = []
        current_chord = None
        
        for segment in harmony_segments:
            chord = segment['chord']
            chord_key = f"{chord['root_name']} {chord['type']}"
            
            if chord_key != current_chord:
                current_chord = chord_key
                progression.append({
                    'root': chord['root'],
                    'root_name': chord['root_name'],
                    'type': chord['type'],
                    'start_time': segment['start_time']
                })
        
        return progression
    
    def get_raga_compatible_chords(self, raga_id):
        """
        Get chords compatible with a specific raga.
        
        Parameters:
        - raga_id: Raga identifier
        
        Returns:
        - Dictionary with compatible chord information
        """
        if raga_id not in self.raga_harmony_mappings:
            return None
        
        return self.raga_harmony_mappings[raga_id]
    
    def suggest_harmony_for_raga(self, raga_id, mood=None):
        """
        Suggest appropriate harmony for a raga based on its characteristics.
        
        Parameters:
        - raga_id: Raga identifier
        - mood: Optional mood to influence harmony suggestions
        
        Returns:
        - Dictionary with harmony suggestions
        """
        if raga_id not in self.raga_harmony_mappings:
            return None
        
        mapping = self.raga_harmony_mappings[raga_id]
        raga = self.ragas.get(raga_id)
        
        # Get raga mood if not specified
        if mood is None and raga:
            mood = raga.get('mood')
        
        # Adjust harmony based on mood
        suggested_progressions = mapping['suggested_progressions']
        
        # Filter or sort progressions based on mood
        if mood == "peaceful":
            # Prefer simpler, consonant progressions
            simple_progs = [prog for prog in suggested_progressions 
                          if len(prog) <= 3]
            if simple_progs:
                suggested_progressions = simple_progs
                
        elif mood == "melancholic":
            # Prefer minor-based progressions
            minor_progs = [prog for prog in suggested_progressions 
                         if any(chord['type'] == 'minor' for chord in prog)]
            if minor_progs:
                suggested_progressions = minor_progs
        
        # Create harmony suggestions
        suggestions = {
            'raga_id': raga_id,
            'raga_name': raga['name'] if raga else None,
            'mood': mood,
            'compatible_chords': mapping['compatible_chords'],
            'suggested_progressions': suggested_progressions,
            'tonic_based_chords': [
                {'root': 0, 'types': mapping['compatible_chords'].get(0, [])},
                {'root': 7, 'types': mapping['compatible_chords'].get(7, [])}
            ]
        }
        
        return suggestions
    
    def visualize_harmony(self, output_file=None):
        """
        Visualize the analyzed harmony.
        
        Parameters:
        - output_file: Path to save visualization image (None for display only)
        
        Returns:
        - True if successful, False otherwise
        """
        if not self.analyzed_harmony:
            print("No harmony analysis available. Run analyze_audio() or analyze_midi() first.")
            return False
        
        # Create figure with multiple subplots
        plt.figure(figsize=(12, 8))
        
        # Plot chromagram
        if 'harmony_segments' in self.analyzed_harmony:
            segments = self.analyzed_harmony['harmony_segments']
            
            # Extract data for plotting
            plt.subplot(2, 1, 1)
            
            # Create a chromagram-like visualization
            if len(segments) > 0 and 'chroma' in segments[0]:
                chroma_data = np.array([segment['chroma'] for segment in segments]).T
                plt.imshow(chroma_data, aspect='auto', origin='lower', 
                         interpolation='nearest', cmap='viridis')
                plt.colorbar(format='%+2.0f dB')
                plt.title('Chroma Features')
                plt.ylabel('Pitch Class')
                plt.xlabel('Segment')
                plt.yticks(range(12), ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
            
            # Plot chord progression
            plt.subplot(2, 1, 2)
            
            if 'chord_progression' in self.analyzed_harmony:
                progression = self.analyzed_harmony['chord_progression']
                
                if progression:
                    chord_labels = [f"{chord['root_name']} {chord['type']}" for chord in progression]
                    
                    x_pos = range(len(chord_labels))
                    plt.bar(x_pos, [1] * len(chord_labels), align='center')
                    plt.xticks(x_pos, chord_labels, rotation=45)
                    plt.xlabel('Chord')
                    plt.ylabel('Presence')
                    plt.title('Chord Progression')
                    plt.tight_layout()
            
            # Save or display
            if output_file:
                plt.savefig(output_file)
                print(f"Visualization saved to {output_file}")
                plt.close()
            else:
                plt.tight_layout()
                plt.show()
            
            return True
        
        return False
    
    def export_results(self, output_file='harmony_analysis.json'):
        """
        Export harmony analysis results to a JSON file.
        
        Parameters:
        - output_file: Output file path
        
        Returns:
        - True if successful, False otherwise
        """
        if not self.analyzed_harmony:
            print("No harmony analysis available. Run analyze_audio() or analyze_midi() first.")
            return False
        
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Convert NumPy arrays to lists for JSON serialization
            export_data = self._prepare_data_for_export(self.analyzed_harmony)
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"Harmony analysis exported to {output_file}")
            return True
            
        except Exception as e:
            print(f"Error exporting harmony analysis: {e}")
            return False
    
    def _prepare_data_for_export(self, data):
        """Prepare data structure for JSON export by converting NumPy arrays to lists."""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, list):
            return [self._prepare_data_for_export(item) for item in data]
        elif isinstance(data, dict):
            return {key: self._prepare_data_for_export(value) for key, value in data.items()}
        else:
            return data


# Example usage when run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze harmony in audio/MIDI files.')
    parser.add_argument('input_file', help='Input audio or MIDI file')
    parser.add_argument('--raga', '-r', help='Raga ID for compatibility analysis')
    parser.add_argument('--output', '-o', help='Output file for results')
    parser.add_argument('--visualize', '-v', action='store_true', help='Visualize harmony')
    
    args = parser.parse_args()
    
    analyzer = HarmonyAnalyzer()
    
    # Detect file type
    if args.input_file.lower().endswith(('.mid', '.midi')):
        results = analyzer.analyze_midi(args.input_file)
    else:
        results = analyzer.analyze_audio(args.input_file)
    
    if results:
        print("\nHarmony Analysis Results:")
        
        if 'predominant_chords' in results:
            print("\nPredominant Chords:")
            for chord, count in results['predominant_chords']:
                print(f"  {chord}: {count}")
        
        if 'chord_progression' in results:
            print("\nChord Progression:")
            for i, chord in enumerate(results['chord_progression']):
                print(f"  {i+1}. {chord['root_name']} {chord['type']}")
        
        # If raga specified, show compatibility
        if args.raga:
            raga_harmony = analyzer.suggest_harmony_for_raga(args.raga)
            
            if raga_harmony:
                print(f"\nHarmony suggestions for raga {raga_harmony['raga_name']}:")
                
                print("\nSuggested Progressions:")
                for i, prog in enumerate(raga_harmony['suggested_progressions']):
                    prog_str = " → ".join([f"{['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][chord['root']]} {chord['type']}" 
                                        for chord in prog])
                    print(f"  {i+1}. {prog_str}")
        
        # Visualize if requested
        if args.visualize:
            analyzer.visualize_harmony(args.output + ".png" if args.output else None)
        
        # Export results if output specified
        if args.output:
            analyzer.export_results(args.output)