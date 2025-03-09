#!/usr/bin/env python3
"""
Carnatic Raga Feature Extractor
------------------------------
Extracts key features that define Carnatic ragas from audio recordings,
including characteristic phrases, gamakas (ornamentations), and note transitions.
"""

import os
import numpy as np
import librosa
import json
import pickle
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d
from audio_analyzer import CarnaticAudioAnalyzer

class RagaFeatureExtractor:
    """
    A class for extracting characteristic features of Carnatic ragas from audio.
    """
    
    def __init__(self, analyzer=None):
        """
        Initialize the feature extractor.
        
        Parameters:
        - analyzer: An existing CarnaticAudioAnalyzer instance, or None to create a new one
        """
        self.analyzer = analyzer if analyzer else CarnaticAudioAnalyzer()
        self.note_events = None
        self.phrases = None
        self.raga_features = {}
        self.raga_models = {}
        
        # Load raga reference models if available
        self._load_raga_models()
    
    def _load_raga_models(self, models_file='data/raga_models.pkl'):
        """Load pre-trained raga models if available."""
        try:
            with open(models_file, 'rb') as f:
                self.raga_models = pickle.load(f)
            print(f"Loaded {len(self.raga_models)} raga models")
        except (FileNotFoundError, pickle.PickleError):
            print("No raga models found. New models will be created from analyzed recordings.")
    
    def analyze_file(self, file_path):
        """
        Analyze an audio file to extract raga features.
        
        Parameters:
        - file_path: Path to the audio file
        
        Returns:
        - Dictionary of extracted raga features
        """
        # Load and analyze the audio file
        if not self.analyzer.load_audio(file_path):
            return None
        
        # Extract pitch data
        self.analyzer.extract_pitch()
        
        # Detect tonic
        self.analyzer.detect_tonic()
        
        # Extract note events
        self.note_events = self.analyzer.extract_note_events()
        
        # Extract musical phrases
        self.phrases = self.analyzer.extract_phrases()
        
        # Extract raga features
        self.raga_features = self._extract_features()
        
        return self.raga_features
    
    def _extract_features(self):
        """
        Extract key features that characterize the raga.
        
        Returns:
        - Dictionary of raga features
        """
        if not self.note_events or not self.phrases:
            print("No note events or phrases available. Run analyze_file() first.")
            return {}
        
        # 1. Extract note distribution
        note_distribution = self._analyze_note_distribution()
        
        # 2. Analyze characteristic phrases
        characteristic_phrases = self._extract_characteristic_phrases()
        
        # 3. Analyze gamakas (ornamentations)
        gamaka_features = self._analyze_gamakas()
        
        # 4. Analyze note transitions
        transition_matrix = self._analyze_note_transitions()
        
        # 5. Analyze phrase structures
        phrase_features = self._analyze_phrase_structures()
        
        # 6. Identify vadi and samvadi (important notes)
        vadi_samvadi = self._identify_vadi_samvadi(note_distribution)
        
        # 7. Identify aarohana and avarohana (ascending/descending patterns)
        arohana_avarohana = self._identify_arohana_avarohana()
        
        # Compile all features
        features = {
            'note_distribution': note_distribution,
            'characteristic_phrases': characteristic_phrases,
            'gamaka_features': gamaka_features,
            'transition_matrix': transition_matrix,
            'phrase_features': phrase_features,
            'vadi_samvadi': vadi_samvadi,
            'arohana_avarohana': arohana_avarohana
        }
        
        return features
    
    def _analyze_note_distribution(self):
        """
        Analyze the distribution of notes relative to the tonic.
        
        Returns:
        - Dictionary with note distribution statistics
        """
        if not self.note_events or not self.analyzer.tonic:
            return {}
        
        # Calculate cents relative to tonic for each note
        tonic_freq = self.analyzer.tonic['frequency']
        note_cents = []
        
        for note in self.note_events:
            cents = 1200 * np.log2(note['frequency'] / tonic_freq)
            # Map to 0-1200 cent range (one octave)
            cents_in_octave = cents % 1200
            note_cents.append(cents_in_octave)
        
        # Create histogram of note distribution
        hist, bin_edges = np.histogram(note_cents, bins=120, range=(0, 1200))
        hist_smoothed = gaussian_filter1d(hist, sigma=2)  # Smooth the histogram
        
        # Find peaks in the histogram (likely scale degrees)
        peak_indices = []
        for i in range(1, len(hist_smoothed) - 1):
            if hist_smoothed[i] > hist_smoothed[i-1] and hist_smoothed[i] > hist_smoothed[i+1]:
                if hist_smoothed[i] > 0.1 * np.max(hist_smoothed):  # Threshold
                    peak_indices.append(i)
        
        peak_cents = [bin_edges[i] for i in peak_indices]
        peak_strengths = [hist_smoothed[i] for i in peak_indices]
        
        # Convert cents to nearest semitone for classification
        scale_degrees = []
        for cents in peak_cents:
            semitone = round(cents / 100)
            scale_degrees.append(semitone % 12)  # Map to one octave
        
        # Count time spent on each note
        note_durations = defaultdict(float)
        for note in self.note_events:
            semitone = round((1200 * np.log2(note['frequency'] / tonic_freq)) / 100) % 12
            note_durations[semitone] += note['duration']
        
        total_duration = sum(note_durations.values())
        note_percentages = {note: (duration / total_duration) * 100 
                            for note, duration in note_durations.items()}
        
        return {
            'scale_degrees': scale_degrees,
            'peak_cents': peak_cents,
            'peak_strengths': peak_strengths,
            'note_durations': dict(note_durations),
            'note_percentages': note_percentages
        }
    
    def _extract_characteristic_phrases(self, min_occurrences=2):
        """
        Extract characteristic melodic phrases from the analyzed recording.
        
        Parameters:
        - min_occurrences: Minimum occurrences for a phrase to be considered characteristic
        
        Returns:
        - List of characteristic phrases with occurrence counts and metadata
        """
        if not self.phrases:
            return []
        
        # Convert note events in phrases to semitones relative to tonic
        tonic_freq = self.analyzer.tonic['frequency']
        phrase_sequences = []
        
        for phrase in self.phrases:
            semitones = []
            for note in phrase:
                cents = 1200 * np.log2(note['frequency'] / tonic_freq)
                semitone = round(cents / 100)
                # Map to one octave for pattern matching
                semitones.append(semitone % 12)
            
            if len(semitones) >= 3:  # Minimum length for a meaningful phrase
                phrase_sequences.append(tuple(semitones))
        
        # Count occurrences of each phrase
        phrase_counts = Counter(phrase_sequences)
        
        # Filter by minimum occurrences
        characteristic = [(phrase, count) for phrase, count in phrase_counts.items() 
                         if count >= min_occurrences]
        
        # Sort by occurrence count (descending)
        characteristic.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to structured representation
        result = []
        for phrase, count in characteristic[:20]:  # Top 20 phrases
            # Calculate average duration of this phrase
            total_duration = 0
            occurrences = 0
            
            for p in self.phrases:
                phrase_semitones = []
                for note in p:
                    cents = 1200 * np.log2(note['frequency'] / tonic_freq)
                    semitone = round(cents / 100) % 12
                    phrase_semitones.append(semitone)
                
                if tuple(phrase_semitones) == phrase:
                    phrase_duration = sum(note['duration'] for note in p)
                    total_duration += phrase_duration
                    occurrences += 1
            
            avg_duration = total_duration / occurrences if occurrences > 0 else 0
            
            result.append({
                'phrase': list(phrase),
                'count': count,
                'avg_duration': avg_duration,
                'context': self._analyze_phrase_context(phrase)
            })
        
        return result
    
    def _analyze_phrase_context(self, phrase):
        """
        Analyze the typical context where a phrase appears.
        
        Parameters:
        - phrase: The phrase to analyze (tuple of semitones)
        
        Returns:
        - Dictionary with context information
        """
        # Check if phrase appears more at the beginning, middle, or end of sections
        positions = []
        phrase_length = len(phrase)
        
        for p in self.phrases:
            phrase_semitones = []
            for note in p:
                cents = 1200 * np.log2(note['frequency'] / self.analyzer.tonic['frequency'])
                semitone = round(cents / 100) % 12
                phrase_semitones.append(semitone)
            
            # Look for the phrase in this sequence
            for i in range(len(phrase_semitones) - phrase_length + 1):
                if tuple(phrase_semitones[i:i+phrase_length]) == phrase:
                    # Calculate relative position (0-1)
                    rel_pos = i / max(1, len(phrase_semitones) - phrase_length)
                    positions.append(rel_pos)
        
        if not positions:
            return {'position': 'unknown'}
        
        avg_position = sum(positions) / len(positions)
        
        # Categorize position
        if avg_position < 0.25:
            position = 'beginning'
        elif avg_position < 0.75:
            position = 'middle'
        else:
            position = 'end'
        
        return {
            'position': position,
            'avg_position': avg_position,
            'count': len(positions)
        }
    
    def _analyze_gamakas(self):
        """
        Analyze gamaka (ornamentation) patterns in the recording.
        
        Returns:
        - Dictionary with gamaka analysis results
        """
        if not self.note_events:
            return {}
        
        # Count notes with gamakas
        gamaka_notes = [note for note in self.note_events if note.get('has_gamaka', False)]
        
        # Calculate percentage of notes with gamakas
        gamaka_percentage = len(gamaka_notes) / len(self.note_events) if self.note_events else 0
        
        # Group gamakas by semitone
        tonic_freq = self.analyzer.tonic['frequency']
        gamakas_by_semitone = defaultdict(list)
        
        for note in gamaka_notes:
            cents = 1200 * np.log2(note['frequency'] / tonic_freq)
            semitone = round(cents / 100) % 12
            gamakas_by_semitone[semitone].append(note)
        
        # Calculate average gamaka intensity by semitone
        gamaka_intensity_by_semitone = {}
        for semitone, notes in gamakas_by_semitone.items():
            avg_intensity = np.mean([note.get('gamaka_intensity', 0) for note in notes])
            gamaka_intensity_by_semitone[semitone] = avg_intensity
        
        # Identify the most frequently ornamented notes
        ornamented_notes = sorted([(semitone, len(notes)) 
                                 for semitone, notes in gamakas_by_semitone.items()],
                                key=lambda x: x[1], reverse=True)
        
        return {
            'gamaka_percentage': gamaka_percentage,
            'ornamented_notes': ornamented_notes,
            'gamaka_intensity_by_semitone': gamaka_intensity_by_semitone,
            'gamaka_notes_count': len(gamaka_notes)
        }
    
    def _analyze_note_transitions(self):
        """
        Analyze transitions between notes to identify melodic patterns.
        
        Returns:
        - Transition probability matrix
        """
        if not self.note_events:
            return {}
        
        # Create transition matrix for semitones (12x12)
        transition_matrix = np.zeros((12, 12))
        
        # Convert notes to semitones relative to tonic
        tonic_freq = self.analyzer.tonic['frequency']
        semitones = []
        
        for note in self.note_events:
            cents = 1200 * np.log2(note['frequency'] / tonic_freq)
            semitone = round(cents / 100) % 12
            semitones.append(semitone)
        
        # Count transitions
        for i in range(len(semitones) - 1):
            from_note = semitones[i]
            to_note = semitones[i + 1]
            transition_matrix[from_note, to_note] += 1
        
        # Convert to probabilities (normalize rows)
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_probs = np.zeros_like(transition_matrix)
        np.divide(transition_matrix, row_sums, out=transition_probs, where=row_sums != 0)
        
        # Convert to Python dictionary
        transition_dict = {}
        for i in range(12):
            transition_dict[i] = {j: transition_probs[i, j] for j in range(12)
                                 if transition_probs[i, j] > 0}
        
        # Identify most common transitions
        common_transitions = []
        for from_note in range(12):
            for to_note in range(12):
                if transition_matrix[from_note, to_note] > 0:
                    common_transitions.append((
                        from_note, 
                        to_note, 
                        transition_matrix[from_note, to_note],
                        transition_probs[from_note, to_note]
                    ))
        
        # Sort by count (descending)
        common_transitions.sort(key=lambda x: x[2], reverse=True)
        
        return {
            'matrix': transition_dict,
            'common_transitions': common_transitions[:20]  # Top 20 transitions
        }
    
    def _analyze_phrase_structures(self):
        """
        Analyze higher-level phrase structures in the music.
        
        Returns:
        - Dictionary with phrase structure analysis
        """
        if not self.phrases:
            return {}
        
        # Calculate phrase durations
        phrase_durations = [sum(note['duration'] for note in phrase) for phrase in self.phrases]
        
        # Calculate phrase complexities (number of notes per second)
        phrase_complexities = [len(phrase) / max(0.1, duration) 
                              for phrase, duration in zip(self.phrases, phrase_durations)]
        
        # Calculate average phrase duration and complexity
        avg_duration = np.mean(phrase_durations) if phrase_durations else 0
        avg_complexity = np.mean(phrase_complexities) if phrase_complexities else 0
        
        # Analyze typical phrase lengths (in number of notes)
        phrase_lengths = [len(phrase) for phrase in self.phrases]
        avg_length = np.mean(phrase_lengths) if phrase_lengths else 0
        
        # Analyze phrase contours (ascending, descending, or mixed)
        phrase_contours = []
        
        for phrase in self.phrases:
            if len(phrase) < 3:
                continue
                
            # Calculate pitch direction for each note transition
            directions = []
            for i in range(len(phrase) - 1):
                if phrase[i+1]['frequency'] > phrase[i]['frequency'] * 1.02:  # 2% threshold
                    directions.append(1)  # Ascending
                elif phrase[i+1]['frequency'] < phrase[i]['frequency'] * 0.98:  # 2% threshold
                    directions.append(-1)  # Descending
                else:
                    directions.append(0)  # Same
            
            # Determine overall contour
            up_count = sum(1 for d in directions if d > 0)
            down_count = sum(1 for d in directions if d < 0)
            
            if up_count > 2 * down_count:
                contour = 'ascending'
            elif down_count > 2 * up_count:
                contour = 'descending'
            elif up_count > down_count:
                contour = 'mostly_ascending'
            elif down_count > up_count:
                contour = 'mostly_descending'
            else:
                contour = 'mixed'
                
            phrase_contours.append(contour)
        
        # Count contour types
        contour_counts = Counter(phrase_contours)
        
        return {
            'avg_duration': avg_duration,
            'avg_complexity': avg_complexity,
            'avg_length': avg_length,
            'phrase_lengths': phrase_lengths,
            'contour_counts': dict(contour_counts)
        }
    
    def _identify_vadi_samvadi(self, note_distribution):
        """
        Identify vadi (most important note) and samvadi (second most important note).
        
        Parameters:
        - note_distribution: Note distribution data
        
        Returns:
        - Dictionary with vadi and samvadi information
        """
        if not note_distribution or 'note_percentages' not in note_distribution:
            return {'vadi': None, 'samvadi': None}
        
        # Get note percentages
        percentages = note_distribution['note_percentages']
        
        if not percentages:
            return {'vadi': None, 'samvadi': None}
        
        # Sort notes by percentage (descending)
        sorted_notes = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
        
        # Find vadi (most important note, excluding tonic)
        vadi = None
        for note, percentage in sorted_notes:
            if note != 0:  # Exclude tonic (Sa)
                vadi = note
                break
        
        # Find samvadi (traditionally a perfect fourth or fifth from vadi)
        samvadi = None
        if vadi is not None:
            # Check if perfect fifth (7 semitones) exists
            fifth = (vadi + 7) % 12
            # Check if perfect fourth (5 semitones) exists
            fourth = (vadi + 5) % 12
            
            if fifth in percentages and fourth in percentages:
                # Choose the one with higher percentage
                if percentages[fifth] > percentages[fourth]:
                    samvadi = fifth
                else:
                    samvadi = fourth
            elif fifth in percentages:
                samvadi = fifth
            elif fourth in percentages:
                samvadi = fourth
            else:
                # If no perfect fourth/fifth, choose second most common note
                for note, percentage in sorted_notes:
                    if note != 0 and note != vadi:  # Exclude tonic and vadi
                        samvadi = note
                        break
        
        return {
            'vadi': vadi,
            'vadi_percentage': percentages.get(vadi, 0) if vadi is not None else 0,
            'samvadi': samvadi,
            'samvadi_percentage': percentages.get(samvadi, 0) if samvadi is not None else 0
        }
    
    def _identify_arohana_avarohana(self):
        """
        Identify the arohana (ascending) and avarohana (descending) patterns.
        
        Returns:
        - Dictionary with arohana and avarohana information
        """
        if not self.phrases:
            return {'arohana': [], 'avarohana': []}
        
        # Find phrases with clear ascending or descending patterns
        ascending_phrases = []
        descending_phrases = []
        
        for phrase in self.phrases:
            if len(phrase) < 4:  # Need a minimum length
                continue
                
            # Convert to semitones relative to tonic
            tonic_freq = self.analyzer.tonic['frequency']
            semitones = []
            
            for note in phrase:
                cents = 1200 * np.log2(note['frequency'] / tonic_freq)
                semitone = round(cents / 100) % 12
                semitones.append(semitone)
            
            # Check if mostly ascending
            is_ascending = True
            for i in range(len(semitones) - 1):
                # Allow some deviation but overall should ascend
                if (semitones[i+1] - semitones[i]) % 12 > 7:  # Wrapped around octave going down
                    is_ascending = False
                    break
                    
            # Check if mostly descending
            is_descending = True
            for i in range(len(semitones) - 1):
                # Allow some deviation but overall should descend
                if 0 < (semitones[i+1] - semitones[i]) % 12 < 7:  # Going up
                    is_descending = False
                    break
            
            if is_ascending:
                ascending_phrases.append(semitones)
            elif is_descending:
                descending_phrases.append(semitones)
        
        # Extract common arohana pattern
        arohana = self._extract_common_scale_pattern(ascending_phrases)
        
        # Extract common avarohana pattern
        avarohana = self._extract_common_scale_pattern(descending_phrases)
        
        return {
            'arohana': arohana,
            'avarohana': avarohana
        }
    
    def _extract_common_scale_pattern(self, phrases):
        """
        Extract a common scale pattern from a set of phrases.
        
        Parameters:
        - phrases: List of phrases (each a list of semitones)
        
        Returns:
        - List representing the common scale pattern
        """
        if not phrases:
            return []
        
        # Count occurrences of each note
        note_counts = Counter()
        for phrase in phrases:
            note_counts.update(phrase)
        
        # Get notes that appear in at least 30% of phrases
        common_notes = set()
        min_count = max(1, len(phrases) * 0.3)
        
        for note, count in note_counts.items():
            if count >= min_count:
                common_notes.add(note)
        
        # Always include Sa (tonic)
        common_notes.add(0)
        
        # Sort notes to create scale pattern
        scale_pattern = sorted(common_notes)
        
        # Ensure the scale has reasonable number of notes (5-9)
        if len(scale_pattern) < 5:
            # Try lowering the threshold
            scale_pattern = self._extract_common_scale_pattern_by_threshold(phrases, 0.2)
        elif len(scale_pattern) > 9:
            # Try raising the threshold
            scale_pattern = self._extract_common_scale_pattern_by_threshold(phrases, 0.4)
        
        return scale_pattern
    
    def _extract_common_scale_pattern_by_threshold(self, phrases, threshold):
        """
        Extract scale pattern using a specific threshold.
        
        Parameters:
        - phrases: List of phrases
        - threshold: Minimum occurrence threshold (0-1)
        
        Returns:
        - List representing the scale pattern
        """
        note_counts = Counter()
        for phrase in phrases:
            note_counts.update(phrase)
        
        common_notes = set()
        min_count = max(1, len(phrases) * threshold)
        
        for note, count in note_counts.items():
            if count >= min_count:
                common_notes.add(note)
        
        # Always include Sa (tonic)
        common_notes.add(0)
        
        return sorted(common_notes)
    
    def identify_raga(self, top_n=3):
        """
        Attempt to identify the raga based on extracted features.
        
        Parameters:
        - top_n: Number of top matches to return
        
        Returns:
        - List of (raga_id, confidence) tuples for top matches
        """
        if not self.raga_features:
            print("No raga features available. Run analyze_file() first.")
            return []
        
        if not self.raga_models:
            print("No raga models available for identification.")
            return []
        
        # Calculate match scores for each raga model
        scores = {}
        
        for raga_id, model in self.raga_models.items():
            # 1. Compare note distributions
            note_score = self._compare_note_distributions(
                self.raga_features['note_distribution'],
                model.get('note_distribution', {})
            )
            
            # 2. Compare arohana/avarohana
            scale_score = self._compare_scale_patterns(
                self.raga_features['arohana_avarohana'],
                model.get('arohana_avarohana', {})
            )
            
            # 3. Compare transition matrices
            transition_score = self._compare_transition_matrices(
                self.raga_features['transition_matrix'],
                model.get('transition_matrix', {})
            )
            
            # 4. Compare gamaka features
            gamaka_score = self._compare_gamaka_features(
                self.raga_features['gamaka_features'],
                model.get('gamaka_features', {})
            )
            
            # 5. Compare phrase features
            phrase_score = self._compare_phrase_features(
                self.raga_features['phrase_features'],
                model.get('phrase_features', {})
            )
            
            # Calculate weighted total score
            total_score = (
                note_score * 0.3 +
                scale_score * 0.3 +
                transition_score * 0.2 +
                gamaka_score * 0.1 +
                phrase_score * 0.1
            )
            
            scores[raga_id] = total_score
        
        # Get top N matches
        top_matches = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return top_matches
    
    def _compare_note_distributions(self, dist1, dist2):
        """Compare two note distributions and return similarity score (0-1)."""
        if not dist1 or not dist2:
            return 0.0
        
        if 'note_percentages' not in dist1 or 'note_percentages' not in dist2:
            return 0.0
        
        # Compare note percentages
        percentages1 = dist1['note_percentages']
        percentages2 = dist2['note_percentages']
        
        # Get union of all notes
        all_notes = set(percentages1.keys()) | set(percentages2.keys())
        
        # Calculate similarity
        similarity = 0.0
        for note in all_notes:
            p1 = percentages1.get(note, 0)
            p2 = percentages2.get(note, 0)
            difference = abs(p1 - p2)
            # Less difference = higher similarity
            similarity += max(0, 100 - difference) / 100
        
        # Normalize
        if all_notes:
            similarity /= len(all_notes)
        
        return similarity
    
    def _compare_scale_patterns(self, patterns1, patterns2):
        """Compare arohana/avarohana patterns and return similarity score (0-1)."""
        if not patterns1 or not patterns2:
            return 0.0
        
        # Compare arohana
        arohana1 = set(patterns1.get('arohana', []))
        arohana2 = set(patterns2.get('arohana', []))
        
        # Compare avarohana
        avarohana1 = set(patterns1.get('avarohana', []))
        avarohana2 = set(patterns2.get('avarohana', []))
        
        # Calculate Jaccard similarity for arohana
        if arohana1 and arohana2:
            arohana_sim = len(arohana1 & arohana2) / len(arohana1 | arohana2)
        else:
            arohana_sim = 0.0
        
        # Calculate Jaccard similarity for avarohana
        if avarohana1 and avarohana2:
            avarohana_sim = len(avarohana1 & avarohana2) / len(avarohana1 | avarohana2)
        else:
            avarohana_sim = 0.0
        
        # Average the similarities
        return (arohana_sim + avarohana_sim) / 2
    
    def _compare_transition_matrices(self, matrix1, matrix2):
        """Compare transition matrices and return similarity score (0-1)."""
        if not matrix1 or not matrix2:
            return 0.0
        
        if 'matrix' not in matrix1 or 'matrix' not in matrix2:
            return 0.0
        
        matrix1 = matrix1['matrix']
        matrix2 = matrix2['matrix']
        
        # Calculate similarity based on common transitions
        similarity = 0.0
        count = 0
        
        for from_note in range(12):
            if str(from_note) in matrix1 and str(from_note) in matrix2:
                for to_note in range(12):
                    to_str = str(to_note)
                    if to_str in matrix1.get(str(from_note), {}) and to_str in matrix2.get(str(from_note), {}):
                        p1 = matrix1[str(from_note)].get(to_str, 0)
                        p2 = matrix2[str(from_note)].get(to_str, 0)
                        difference = abs(p1 - p2)
                        # Less difference = higher similarity
                        similarity += max(0, 1 - difference)
                        count += 1
        
        # Normalize
        if count > 0:
            similarity /= count
        
        return similarity
    
    def _compare_gamaka_features(self, gamaka1, gamaka2):
        """Compare gamaka features and return similarity score (0-1)."""
        if not gamaka1 or not gamaka2:
            return 0.0
        
        # Compare overall gamaka percentage
        if 'gamaka_percentage' in gamaka1 and 'gamaka_percentage' in gamaka2:
            p1 = gamaka1['gamaka_percentage']
            p2 = gamaka2['gamaka_percentage']
            percentage_sim = max(0, 1 - abs(p1 - p2))
        else:
            percentage_sim = 0.0
        
        # Compare ornamented notes
        ornamented1 = set(note for note, _ in gamaka1.get('ornamented_notes', []))
        ornamented2 = set(note for note, _ in gamaka2.get('ornamented_notes', []))
        
        if ornamented1 and ornamented2:
            notes_sim = len(ornamented1 & ornamented2) / len(ornamented1 | ornamented2)
        else:
            notes_sim = 0.0
        
        # Weighted average
        return percentage_sim * 0.4 + notes_sim * 0.6
    
    def _compare_phrase_features(self, phrase1, phrase2):
        """Compare phrase features and return similarity score (0-1)."""
        if not phrase1 or not phrase2:
            return 0.0
        
        # Compare average phrase length
        if 'avg_length' in phrase1 and 'avg_length' in phrase2:
            length1 = phrase1['avg_length']
            length2 = phrase2['avg_length']
            length_diff = abs(length1 - length2)
            length_sim = max(0, 1 - (length_diff / max(length1, length2, 1)))
        else:
            length_sim = 0.0
        
        # Compare contour counts
        if 'contour_counts' in phrase1 and 'contour_counts' in phrase2:
            contour1 = phrase1['contour_counts']
            contour2 = phrase2['contour_counts']
            
            # Calculate similarity of contour distributions
            all_contours = set(contour1.keys()) | set(contour2.keys())
            
            if all_contours:
                contour_sim = 0.0
                for contour in all_contours:
                    c1 = contour1.get(contour, 0)
                    c2 = contour2.get(contour, 0)
                    # Normalize counts to percentages
                    total1 = sum(contour1.values()) or 1
                    total2 = sum(contour2.values()) or 1
                    p1 = c1 / total1
                    p2 = c2 / total2
                    contour_sim += min(p1, p2)  # Common percentage
                
                contour_sim = min(1.0, contour_sim)  # Cap at 1.0
            else:
                contour_sim = 0.0
        else:
            contour_sim = 0.0
        
        # Weighted average
        return length_sim * 0.4 + contour_sim * 0.6
    
    def update_raga_model(self, raga_id, raga_name=None):
        """
        Update or create a raga model based on the current analysis.
        
        Parameters:
        - raga_id: ID of the raga
        - raga_name: Name of the raga (optional)
        
        Returns:
        - True if successful, False otherwise
        """
        if not self.raga_features:
            print("No raga features available. Run analyze_file() first.")
            return False
        
        # Create new model or update existing one
        if raga_id in self.raga_models:
            model = self.raga_models[raga_id]
            # Update with new data (simple averaging)
            model = self._merge_raga_models(model, self.raga_features)
        else:
            # Create new model
            model = self.raga_features.copy()
            if raga_name:
                model['name'] = raga_name
        
        # Update the models dictionary
        self.raga_models[raga_id] = model
        
        return True
    
    def _merge_raga_models(self, model1, model2, weight2=0.3):
        """
        Merge two raga models, giving weight to the new model.
        
        Parameters:
        - model1: Existing model
        - model2: New model to merge in
        - weight2: Weight for the new model (0-1)
        
        Returns:
        - Merged model
        """
        # Simple dictionary for merged model
        merged = {}
        
        # Keep the name from the existing model
        if 'name' in model1:
            merged['name'] = model1['name']
        
        # Merge note distributions
        if 'note_distribution' in model1 and 'note_distribution' in model2:
            merged['note_distribution'] = self._merge_note_distributions(
                model1['note_distribution'],
                model2['note_distribution'],
                weight2
            )
        
        # Merge arohana/avarohana
        if 'arohana_avarohana' in model1 and 'arohana_avarohana' in model2:
            merged['arohana_avarohana'] = self._merge_arohana_avarohana(
                model1['arohana_avarohana'],
                model2['arohana_avarohana']
            )
        
        # Merge transition matrices
        if 'transition_matrix' in model1 and 'transition_matrix' in model2:
            merged['transition_matrix'] = self._merge_transition_matrices(
                model1['transition_matrix'],
                model2['transition_matrix'],
                weight2
            )
        
        # Merge gamaka features
        if 'gamaka_features' in model1 and 'gamaka_features' in model2:
            merged['gamaka_features'] = self._merge_gamaka_features(
                model1['gamaka_features'],
                model2['gamaka_features'],
                weight2
            )
        
        # Merge phrase features
        if 'phrase_features' in model1 and 'phrase_features' in model2:
            merged['phrase_features'] = self._merge_phrase_features(
                model1['phrase_features'],
                model2['phrase_features'],
                weight2
            )
        
        # Merge characteristic phrases
        if 'characteristic_phrases' in model1 and 'characteristic_phrases' in model2:
            merged['characteristic_phrases'] = self._merge_characteristic_phrases(
                model1['characteristic_phrases'],
                model2['characteristic_phrases']
            )
        
        # Merge vadi/samvadi
        if 'vadi_samvadi' in model1 and 'vadi_samvadi' in model2:
            merged['vadi_samvadi'] = self._merge_vadi_samvadi(
                model1['vadi_samvadi'],
                model2['vadi_samvadi']
            )
        
        return merged
    
    def _merge_note_distributions(self, dist1, dist2, weight2=0.3):
        """Merge two note distributions."""
        merged = {}
        
        # Merge note percentages
        if 'note_percentages' in dist1 and 'note_percentages' in dist2:
            percentages1 = dist1['note_percentages']
            percentages2 = dist2['note_percentages']
            
            merged_percentages = {}
            all_notes = set(percentages1.keys()) | set(percentages2.keys())
            
            for note in all_notes:
                p1 = percentages1.get(note, 0)
                p2 = percentages2.get(note, 0)
                # Weighted average
                merged_percentages[note] = p1 * (1 - weight2) + p2 * weight2
            
            merged['note_percentages'] = merged_percentages
        
        # Merge scale degrees (take the union)
        if 'scale_degrees' in dist1 and 'scale_degrees' in dist2:
            merged['scale_degrees'] = sorted(set(dist1['scale_degrees']) | set(dist2['scale_degrees']))
        
        return merged
    
    def _merge_arohana_avarohana(self, aa1, aa2):
        """Merge two sets of arohana/avarohana patterns."""
        merged = {}
        
        # For arohana and avarohana, take the union of notes
        if 'arohana' in aa1 and 'arohana' in aa2:
            merged['arohana'] = sorted(set(aa1['arohana']) | set(aa2['arohana']))
        
        if 'avarohana' in aa1 and 'avarohana' in aa2:
            merged['avarohana'] = sorted(set(aa1['avarohana']) | set(aa2['avarohana']))
        
        return merged
    
    def _merge_transition_matrices(self, tm1, tm2, weight2=0.3):
        """Merge two transition matrices."""
        merged = {}
        
        # Merge matrices
        if 'matrix' in tm1 and 'matrix' in tm2:
            matrix1 = tm1['matrix']
            matrix2 = tm2['matrix']
            
            merged_matrix = {}
            all_from_notes = set(matrix1.keys()) | set(matrix2.keys())
            
            for from_note in all_from_notes:
                merged_matrix[from_note] = {}
                
                # Get transitions from both models
                transitions1 = matrix1.get(from_note, {})
                transitions2 = matrix2.get(from_note, {})
                
                all_to_notes = set(transitions1.keys()) | set(transitions2.keys())
                
                for to_note in all_to_notes:
                    p1 = transitions1.get(to_note, 0)
                    p2 = transitions2.get(to_note, 0)
                    # Weighted average
                    merged_matrix[from_note][to_note] = p1 * (1 - weight2) + p2 * weight2
            
            merged['matrix'] = merged_matrix
        
        # Take top transitions from both models
        common_transitions = []
        
        if 'common_transitions' in tm1:
            common_transitions.extend(tm1['common_transitions'])
        
        if 'common_transitions' in tm2:
            common_transitions.extend(tm2['common_transitions'])
        
        # Sort and deduplicate
        unique_transitions = {}
        for from_note, to_note, count, prob in common_transitions:
            key = (from_note, to_note)
            if key not in unique_transitions or count > unique_transitions[key][0]:
                unique_transitions[key] = (count, prob)
        
        merged_transitions = []
        for (from_note, to_note), (count, prob) in unique_transitions.items():
            merged_transitions.append((from_note, to_note, count, prob))
        
        # Sort by count (descending)
        merged_transitions.sort(key=lambda x: x[2], reverse=True)
        
        merged['common_transitions'] = merged_transitions[:20]  # Top 20
        
        return merged
    
    def _merge_gamaka_features(self, gf1, gf2, weight2=0.3):
        """Merge two sets of gamaka features."""
        merged = {}
        
        # Merge gamaka percentage
        if 'gamaka_percentage' in gf1 and 'gamaka_percentage' in gf2:
            p1 = gf1['gamaka_percentage']
            p2 = gf2['gamaka_percentage']
            merged['gamaka_percentage'] = p1 * (1 - weight2) + p2 * weight2
        
        # Merge ornamented notes
        ornamented_notes = {}
        
        for note, count in gf1.get('ornamented_notes', []):
            ornamented_notes[note] = ornamented_notes.get(note, 0) + count * (1 - weight2)
        
        for note, count in gf2.get('ornamented_notes', []):
            ornamented_notes[note] = ornamented_notes.get(note, 0) + count * weight2
        
        merged_ornamented = [(note, count) for note, count in ornamented_notes.items()]
        merged_ornamented.sort(key=lambda x: x[1], reverse=True)
        
        merged['ornamented_notes'] = merged_ornamented
        
        # Merge gamaka intensity by semitone
        intensity_by_semitone = {}
        
        for semitone, intensity in gf1.get('gamaka_intensity_by_semitone', {}).items():
            intensity_by_semitone[semitone] = intensity_by_semitone.get(semitone, 0) + intensity * (1 - weight2)
        
        for semitone, intensity in gf2.get('gamaka_intensity_by_semitone', {}).items():
            intensity_by_semitone[semitone] = intensity_by_semitone.get(semitone, 0) + intensity * weight2
        
        merged['gamaka_intensity_by_semitone'] = intensity_by_semitone
        
        return merged
    
    def _merge_phrase_features(self, pf1, pf2, weight2=0.3):
        """Merge two sets of phrase features."""
        merged = {}
        
        # Merge average values
        for key in ['avg_duration', 'avg_complexity', 'avg_length']:
            if key in pf1 and key in pf2:
                v1 = pf1[key]
                v2 = pf2[key]
                merged[key] = v1 * (1 - weight2) + v2 * weight2
        
        # Combine phrase lengths
        if 'phrase_lengths' in pf1 and 'phrase_lengths' in pf2:
            merged['phrase_lengths'] = pf1['phrase_lengths'] + pf2['phrase_lengths']
        
        # Merge contour counts
        contour_counts = {}
        
        for contour, count in pf1.get('contour_counts', {}).items():
            contour_counts[contour] = contour_counts.get(contour, 0) + count * (1 - weight2)
        
        for contour, count in pf2.get('contour_counts', {}).items():
            contour_counts[contour] = contour_counts.get(contour, 0) + count * weight2
        
        merged['contour_counts'] = contour_counts
        
        return merged
    
    def _merge_characteristic_phrases(self, cp1, cp2):
        """Merge two sets of characteristic phrases."""
        # Combine all phrases
        all_phrases = {}
        
        for phrase_data in cp1:
            phrase_tuple = tuple(phrase_data['phrase'])
            if phrase_tuple not in all_phrases or phrase_data['count'] > all_phrases[phrase_tuple]['count']:
                all_phrases[phrase_tuple] = phrase_data
        
        for phrase_data in cp2:
            phrase_tuple = tuple(phrase_data['phrase'])
            if phrase_tuple not in all_phrases or phrase_data['count'] > all_phrases[phrase_tuple]['count']:
                all_phrases[phrase_tuple] = phrase_data
        
        # Convert back to list
        merged_phrases = list(all_phrases.values())
        
        # Sort by count (descending)
        merged_phrases.sort(key=lambda x: x['count'], reverse=True)
        
        return merged_phrases[:20]  # Top 20
    
    def _merge_vadi_samvadi(self, vs1, vs2):
        """Merge two sets of vadi/samvadi information."""
        # If they agree, use the common values
        if vs1['vadi'] == vs2['vadi']:
            return vs1
        elif vs1.get('vadi_percentage', 0) >= vs2.get('vadi_percentage', 0):
            return vs1
        else:
            return vs2
    
    def save_raga_models(self, filename='data/raga_models.pkl'):
        """
        Save all raga models to a file.
        
        Parameters:
        - filename: Output filename
        
        Returns:
        - True if successful, False otherwise
        """
        if not self.raga_models:
            print("No raga models to save.")
            return False
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.raga_models, f)
            
            print(f"Saved {len(self.raga_models)} raga models to {filename}")
            return True
        except Exception as e:
            print(f"Error saving raga models: {e}")
            return False
    
    def export_raga_features_json(self, filename='outputs/raga_features.json', raga_id=None, raga_name=None):
        """
        Export raga features to a JSON file for use in other modules.
        
        Parameters:
        - filename: Output filename
        - raga_id: ID of the raga
        - raga_name: Name of the raga
        
        Returns:
        - True if successful, False otherwise
        """
        if not self.raga_features:
            print("No raga features to export. Run analyze_file() first.")
            return False
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Prepare export data
        export_data = self.raga_features.copy()
        
        # Add metadata
        export_data['metadata'] = {
            'raga_id': raga_id,
            'raga_name': raga_name,
            'tonic': self.analyzer.tonic['note'] if self.analyzer.tonic else None,
            'tonic_frequency': self.analyzer.tonic['frequency'] if self.analyzer.tonic else None
        }
        
        # Convert NumPy arrays and other non-serializable types
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.number):
                return float(obj)
            elif isinstance(obj, dict):
                return {str(k): convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, tuple):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        export_data = convert_for_json(export_data)
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"Raga features exported to {filename}")
            return True
        except Exception as e:
            print(f"Error exporting raga features: {e}")
            return False
    
    def plot_raga_features(self):
        """
        Plot visualizations of the extracted raga features.
        
        Returns:
        - True if successful, False otherwise
        """
        if not self.raga_features:
            print("No raga features to plot. Run analyze_file() first.")
            return False
        
        # Create figure with multiple subplots
        plt.figure(figsize=(15, 12))
        
        # 1. Plot note distribution
        plt.subplot(2, 2, 1)
        self._plot_note_distribution()
        
        # 2. Plot transition matrix as heatmap
        plt.subplot(2, 2, 2)
        self._plot_transition_matrix()
        
        # 3. Plot gamaka features
        plt.subplot(2, 2, 3)
        self._plot_gamaka_features()
        
        # 4. Plot phrase features
        plt.subplot(2, 2, 4)
        self._plot_phrase_features()
        
        plt.tight_layout()
        plt.show()
        
        return True
    
    def _plot_note_distribution(self):
        """Plot the note distribution."""
        note_dist = self.raga_features.get('note_distribution', {})
        
        if not note_dist or 'note_percentages' not in note_dist:
            plt.title("Note Distribution (No Data)")
            return
        
        percentages = note_dist['note_percentages']
        
        # Convert to ordered list
        notes = []
        values = []
        
        # Use Indian note names
        indian_notes = ['Sa', 'Komal Re', 'Re', 'Komal Ga', 'Ga', 'Ma', 'Tivra Ma', 
                       'Pa', 'Komal Dha', 'Dha', 'Komal Ni', 'Ni']
        
        for note in range(12):
            notes.append(indian_notes[note])
            values.append(percentages.get(note, 0))
        
        # Plot
        plt.bar(notes, values)
        plt.title("Note Distribution")
        plt.xlabel("Note")
        plt.ylabel("Percentage (%)")
        plt.xticks(rotation=45)
        
        # Highlight vadi and samvadi
        vadi_samvadi = self.raga_features.get('vadi_samvadi', {})
        
        if 'vadi' in vadi_samvadi and vadi_samvadi['vadi'] is not None:
            vadi = vadi_samvadi['vadi']
            plt.bar(indian_notes[vadi], percentages.get(vadi, 0), color='red', 
                   label=f'Vadi: {indian_notes[vadi]}')
        
        if 'samvadi' in vadi_samvadi and vadi_samvadi['samvadi'] is not None:
            samvadi = vadi_samvadi['samvadi']
            plt.bar(indian_notes[samvadi], percentages.get(samvadi, 0), color='green', 
                   label=f'Samvadi: {indian_notes[samvadi]}')
        
        plt.legend()
    
    def _plot_transition_matrix(self):
        """Plot the transition matrix as a heatmap."""
        transition_data = self.raga_features.get('transition_matrix', {})
        
        if not transition_data or 'matrix' not in transition_data:
            plt.title("Transition Matrix (No Data)")
            return
        
        matrix_dict = transition_data['matrix']
        
        # Convert dictionary to matrix
        matrix = np.zeros((12, 12))
        
        for from_note, transitions in matrix_dict.items():
            for to_note, prob in transitions.items():
                matrix[int(from_note), int(to_note)] = prob
        
        # Plot heatmap
        plt.imshow(matrix, cmap='Blues')
        plt.colorbar(label='Transition Probability')
        plt.title("Note Transition Matrix")
        
        # Use Indian note names for labels
        indian_notes = ['Sa', 'r', 'R', 'g', 'G', 'm', 'M', 'P', 'd', 'D', 'n', 'N']
        
        plt.xticks(range(12), indian_notes)
        plt.yticks(range(12), indian_notes)
        plt.xlabel("To Note")
        plt.ylabel("From Note")
    
    def _plot_gamaka_features(self):
        """Plot gamaka features."""
        gamaka_features = self.raga_features.get('gamaka_features', {})
        
        if not gamaka_features:
            plt.title("Gamaka Features (No Data)")
            return
        
        # Get ornamented notes
        ornamented_notes = gamaka_features.get('ornamented_notes', [])
        
        if not ornamented_notes:
            plt.title("Gamaka Features (No Data)")
            return
        
        # Extract data for plotting
        notes = []
        counts = []
        
        # Use Indian note names
        indian_notes = ['Sa', 'r', 'R', 'g', 'G', 'm', 'M', 'P', 'd', 'D', 'n', 'N']
        
        for note, count in ornamented_notes[:8]:  # Top 8 for clarity
            notes.append(indian_notes[note])
            counts.append(count)
        
        # Plot
        bars = plt.bar(notes, counts)
        plt.title(f"Ornamented Notes (Gamaka %: {gamaka_features.get('gamaka_percentage', 0):.1%})")
        plt.xlabel("Note")
        plt.ylabel("Ornamentation Count")
        
        # Add intensity as text on bars
        intensity_by_semitone = gamaka_features.get('gamaka_intensity_by_semitone', {})
        
        for i, note in enumerate(ornamented_notes[:8]):
            note_val = note[0]
            if note_val in intensity_by_semitone:
                plt.text(i, counts[i] * 0.5, f"{intensity_by_semitone[note_val]:.2f}", 
                        ha='center', color='white', fontweight='bold')
    
    def _plot_phrase_features(self):
        """Plot phrase features."""
        phrase_features = self.raga_features.get('phrase_features', {})
        
        if not phrase_features or 'contour_counts' not in phrase_features:
            plt.title("Phrase Features (No Data)")
            return
        
        # Get contour counts
        contour_counts = phrase_features['contour_counts']
        
        # Extract data for plotting
        contours = []
        counts = []
        
        for contour, count in contour_counts.items():
            contours.append(contour)
            counts.append(count)
        
        # Plot
        plt.bar(contours, counts)
        plt.title(f"Phrase Contours (Avg Length: {phrase_features.get('avg_length', 0):.1f} notes)")
        plt.xlabel("Contour Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        
        # Add avg duration and complexity as text
        avg_duration = phrase_features.get('avg_duration', 0)
        avg_complexity = phrase_features.get('avg_complexity', 0)
        
        plt.annotate(f"Avg Duration: {avg_duration:.2f}s\nAvg Complexity: {avg_complexity:.2f} notes/s",
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))


def analyze_raga_features(file_path, output_dir='outputs', raga_id=None, raga_name=None, plot=False):
    """
    Analyze raga features from an audio file and export the results.
    
    Parameters:
    - file_path: Path to the audio file
    - output_dir: Directory for output files
    - raga_id: ID of the raga (if known)
    - raga_name: Name of the raga (if known)
    - plot: Whether to display plots during analysis
    
    Returns:
    - Dictionary of raga features
    """
    # Create analyzer
    analyzer = CarnaticAudioAnalyzer()
    extractor = RagaFeatureExtractor(analyzer)
    
    # Analyze file
    features = extractor.analyze_file(file_path)
    
    if not features:
        print("Failed to analyze raga features.")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get filename without extension
    filename = os.path.splitext(os.path.basename(file_path))[0]
    
    # Export features
    extractor.export_raga_features_json(
        os.path.join(output_dir, f"{filename}_raga_features.json"),
        raga_id,
        raga_name
    )
    
    # Plot features if requested
    if plot:
        extractor.plot_raga_features()
    
    # If raga_id is provided, update the model
    if raga_id:
        extractor.update_raga_model(raga_id, raga_name)
        extractor.save_raga_models()
    
    # Try to identify the raga if not specified
    if not raga_id and extractor.raga_models:
        identified_ragas = extractor.identify_raga()
        if identified_ragas:
            top_raga_id, confidence = identified_ragas[0]
            print(f"\nIdentified raga: {top_raga_id} (Confidence: {confidence:.2f})")
            for raga_id, conf in identified_ragas[1:3]:
                print(f"Alternative match: {raga_id} (Confidence: {conf:.2f})")
    
    return features


# Example usage
if __name__ == "__main__":
    # Check if file path is provided
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        raga_id = sys.argv[2] if len(sys.argv) > 2 else None
        raga_name = sys.argv[3] if len(sys.argv) > 3 else None
        
        features = analyze_raga_features(file_path, raga_id=raga_id, raga_name=raga_name, plot=True)
        
        if features:
            print("\nRaga Features Summary:")
            
            # Print vadi and samvadi
            vadi_samvadi = features.get('vadi_samvadi', {})
            if vadi_samvadi:
                indian_notes = ['Sa', 'Komal Re', 'Re', 'Komal Ga', 'Ga', 'Ma', 'Tivra Ma', 
                               'Pa', 'Komal Dha', 'Dha', 'Komal Ni', 'Ni']
                
                vadi = vadi_samvadi.get('vadi')
                samvadi = vadi_samvadi.get('samvadi')
                
                if vadi is not None:
                    print(f"  Vadi (dominant note): {indian_notes[vadi]}")
                
                if samvadi is not None:
                    print(f"  Samvadi (second dominant): {indian_notes[samvadi]}")
            
            # Print arohana and avarohana
            arohana_avarohana = features.get('arohana_avarohana', {})
            if arohana_avarohana:
                indian_notes = ['Sa', 'r', 'R', 'g', 'G', 'm', 'M', 'P', 'd', 'D', 'n', 'N']
                
                arohana = arohana_avarohana.get('arohana', [])
                avarohana = arohana_avarohana.get('avarohana', [])
                
                if arohana:
                    arohana_str = ' '.join(indian_notes[n] for n in arohana)
                    print(f"  Arohana (ascending): {arohana_str}")
                
                if avarohana:
                    avarohana_str = ' '.join(indian_notes[n] for n in avarohana)
                    print(f"  Avarohana (descending): {avarohana_str}")
            
            # Print gamaka percentage
            gamaka_features = features.get('gamaka_features', {})
            if gamaka_features:
                print(f"  Gamaka percentage: {gamaka_features.get('gamaka_percentage', 0):.1%}")
    else:
        print("Please provide an audio file path to analyze.")
        print("Usage: python raga_feature_extractor.py <audio_file_path> [raga_id] [raga_name]")