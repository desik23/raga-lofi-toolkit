#!/usr/bin/env python3
"""
Audio Analyzer for Carnatic Music
---------------------------------
Analyzes Carnatic music recordings to extract melodic patterns,
identify ragas, and detect characteristic gamakas.
"""

import os
import numpy as np
import librosa
import librosa.display
import json
import pickle
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

class CarnaticAudioAnalyzer:
    """
    A class for analyzing Carnatic music recordings and extracting musical features.
    """
    
    def __init__(self, sample_rate=22050):
        """
        Initialize the analyzer.
        
        Parameters:
        - sample_rate: Sample rate to use for analysis
        """
        self.sample_rate = sample_rate
        self.pitch_data = None
        self.tonic = None
        self.note_events = None
        
        # Load raga information if available
        self.ragas = {}
        self._load_raga_info()
    
    def _load_raga_info(self):
        """Load raga information from the ragas.json file."""
        try:
            with open('data/ragas.json', 'r') as f:
                data = json.load(f)
                # Create a dictionary of raga information
                for raga in data['ragas']:
                    self.ragas[raga['id']] = raga
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load raga information: {e}")
    
    def load_audio(self, file_path):
        """
        Load an audio file for analysis.
        
        Parameters:
        - file_path: Path to the audio file
        
        Returns:
        - True if successful, False otherwise
        """
        try:
            # Load the audio file
            y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            self.audio = y
            self.duration = librosa.get_duration(y=y, sr=sr)
            
            print(f"Loaded audio file: {file_path}")
            print(f"Duration: {self.duration:.2f} seconds")
            
            # Reset analysis data
            self.pitch_data = None
            self.tonic = None
            self.note_events = None
            
            return True
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return False
    
    def extract_pitch(self, hop_length=512, fmin=80, fmax=1000, filter_size=15):
        """
        Extract pitch (fundamental frequency) from the audio.
        
        Parameters:
        - hop_length: Hop length for pitch tracking
        - fmin: Minimum frequency to consider
        - fmax: Maximum frequency to consider
        - filter_size: Size of median filter for smoothing
        
        Returns:
        - Tuple of (times, frequencies, confidence)
        """
        if not hasattr(self, 'audio'):
            print("No audio loaded. Please call load_audio() first.")
            return None, None, None
        
        # Extract pitch using pYIN algorithm
        f0, voiced_flag, voiced_probs = librosa.pyin(
            self.audio, 
            fmin=fmin,
            fmax=fmax,
            sr=self.sample_rate,
            hop_length=hop_length
        )
        
        # Apply median filter to remove noise
        if filter_size > 0:
            f0_smooth = medfilt(f0, filter_size)
            # Replace NaN values
            f0_smooth[np.isnan(f0_smooth)] = 0
        else:
            f0_smooth = f0
            # Replace NaN values
            f0_smooth[np.isnan(f0_smooth)] = 0
        
        # Calculate time points
        times = librosa.times_like(f0, sr=self.sample_rate, hop_length=hop_length)
        
        # Store results
        self.pitch_data = {
            'times': times,
            'frequencies': f0_smooth,
            'confidence': voiced_probs
        }
        
        print(f"Extracted pitch data: {len(times)} frames")
        
        return times, f0_smooth, voiced_probs
    
    def detect_tonic(self, plot=False):
        """
        Detect the tonic (Sa) of the performance.
        
        Parameters:
        - plot: Whether to plot the pitch histogram
        
        Returns:
        - Detected tonic frequency
        """
        if self.pitch_data is None:
            print("No pitch data available. Please run extract_pitch() first.")
            return None
        
        # Filter out unvoiced and zero frequencies
        frequencies = self.pitch_data['frequencies']
        voiced = (frequencies > 0) & (~np.isnan(frequencies))
        valid_frequencies = frequencies[voiced]
        
        if len(valid_frequencies) == 0:
            print("No valid frequencies found for tonic detection.")
            return None
        
        # Convert frequencies to cents relative to C1 (to create pitch histogram)
        cents = 1200 * np.log2(valid_frequencies / librosa.note_to_hz('C1'))
        
        # Create pitch histogram
        bins = np.linspace(0, 7200, 720)  # 720 bins spanning 6 octaves from C1
        hist, bin_edges = np.histogram(cents, bins=bins)
        
        # Smooth the histogram
        hist_smooth = np.convolve(hist, np.hanning(10) / np.sum(np.hanning(10)), mode='same')
        
        # Find peaks in the histogram
        peak_indices = librosa.util.peak_pick(hist_smooth, pre_max=20, post_max=20, 
                                              pre_avg=50, post_avg=50, delta=0.5, wait=10)
        peak_values = hist_smooth[peak_indices]
        
        # Sort peaks by value (descending)
        sorted_indices = np.argsort(peak_values)[::-1]
        peak_indices = peak_indices[sorted_indices]
        peak_bin_centers = (bin_edges[peak_indices] + bin_edges[peak_indices + 1]) / 2
        
        # Check if we found any peaks
        if len(peak_indices) == 0:
            print("No clear peaks found in pitch histogram.")
            return None
        
        # The highest peak is likely the tonic or a dominant note
        # For Carnatic music, we need to check musical context as well
        tonic_candidates = []
        
        # Convert peaks back to frequency
        for cent in peak_bin_centers[:5]:  # Consider top 5 peaks
            freq = librosa.hz_to_note(librosa.midi_to_hz(cent / 100 + 24))
            tonic_candidates.append((freq, cent))
        
        # For now, use the most prominent peak as tonic
        tonic_cents = peak_bin_centers[0]
        tonic_freq = librosa.midi_to_hz(tonic_cents / 100 + 24)
        tonic_note = librosa.hz_to_note(tonic_freq)
        
        # Store the detected tonic
        self.tonic = {
            'frequency': tonic_freq,
            'note': tonic_note,
            'cents': tonic_cents
        }
        
        print(f"Detected tonic: {tonic_note} ({tonic_freq:.2f} Hz)")
        print(f"Tonic candidates: {tonic_candidates}")
                # Add this stabilization code at the end
        if hasattr(self, 'previous_tonic') and self.previous_tonic:
            # Check if the new tonic is very different from previous detection
            prev_freq = self.previous_tonic['frequency']
            new_freq = tonic_freq
            
            # Calculate ratio - should be close to 1.0 or a simple multiple
            ratio = new_freq / prev_freq
            
            # If the ratio is close to a semitone difference or greater, be suspicious
            if abs(1.0 - ratio) < 0.03:
                # Very close match, keep new tonic
                pass
            elif any(abs(ratio - r) < 0.05 for r in [2.0, 0.5]):
                # Octave difference, adjust to previous octave
                print(f"Adjusting tonic octave to match previous detection")
                if ratio > 1:
                    tonic_freq = tonic_freq / 2
                else:
                    tonic_freq = tonic_freq * 2
                tonic_note = librosa.hz_to_note(tonic_freq)
            else:
                # Significant difference - use the previous tonic if confidence is low
                confidence_threshold = 0.7  # Adjust as needed
                if top_peak_value / total_peak_weight < confidence_threshold:
                    print(f"Low confidence tonic detection ({tonic_note}), reverting to previous tonic ({self.previous_tonic['note']})")
                    tonic_freq = prev_freq
                    tonic_note = self.previous_tonic['note']
        
        # Store the detected tonic for future reference
        self.previous_tonic = {
            'frequency': tonic_freq,
            'note': tonic_note,
            'cents': tonic_cents
        }
        
        self.tonic = self.previous_tonic
        # Plot histogram if requested
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(bin_edges[:-1], hist_smooth)
            plt.vlines(peak_bin_centers[:5], 0, np.max(hist_smooth), color='r', linestyle='--')
            plt.xlabel('Cents (relative to C1)')
            plt.ylabel('Count')
            plt.title('Pitch Histogram with Tonic Candidates')
            plt.tight_layout()
            plt.show()
        
        return tonic_freq
    
    def extract_note_events(self, min_duration=0.05, min_confidence=0.2):  # Reduced thresholds
        """
        Extract note events from pitch data.
        
        Parameters:
        - min_duration: Minimum duration of a note (in seconds)
        - min_confidence: Minimum confidence level for pitch detection
        
        Returns:
        - List of note events with timing and frequency information
        """
        if self.pitch_data is None:
            print("No pitch data available. Please run extract_pitch() first.")
            return None
        
        if self.tonic is None:
            print("No tonic detected. Please run detect_tonic() first.")
            return None
        
        times = self.pitch_data['times']
        frequencies = self.pitch_data['frequencies']
        confidence = self.pitch_data['confidence']
        
        # Step 1: Add more aggressive filtering for valid frequencies
        # Use a lower threshold for valid frequencies
        valid_mask = (frequencies > 30) & (confidence > min_confidence)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) < 5:  # Need some minimum number of valid frames
            print(f"Too few valid frequencies ({len(valid_indices)}) for note extraction.")
            # Create at least one note event from the maximum confidence segment
            if len(confidence) > 0:
                max_conf_idx = np.argmax(confidence)
                if confidence[max_conf_idx] > 0 and frequencies[max_conf_idx] > 30:
                    # Create a single note event
                    note_freq = frequencies[max_conf_idx]
                    cents_from_tonic = 1200 * np.log2(note_freq / max(1.0, self.tonic['frequency']))
                    semitone = round(cents_from_tonic / 100)
                    
                    note_events = [{
                        'start_time': times[0],
                        'end_time': times[-1],
                        'duration': times[-1] - times[0],
                        'frequency': note_freq,
                        'semitone': semitone,
                        'deviation': 0,
                        'confidence': confidence[max_conf_idx],
                        'has_gamaka': False,
                        'gamaka_intensity': 0
                    }]
                    
                    self.note_events = note_events
                    print(f"Created one note event as fallback")
                    return note_events
            
            return []
        
        # Step 2: Work only with valid segments
        filtered_times = times[valid_indices]
        filtered_freqs = frequencies[valid_indices]
        filtered_conf = confidence[valid_indices]
        
        # Create a new semitones array with only valid values
        cents_relative_to_tonic = 1200 * np.log2(filtered_freqs / max(1.0, self.tonic['frequency']))
        semitones = np.round(cents_relative_to_tonic / 100)
        
        # Step 3: Segment notes based on changes or pauses
        # Find where semitones change
        semitone_changes = np.where(np.diff(semitones) != 0)[0]
        
        # Also find where there might be pauses
        time_diffs = np.diff(filtered_times)
        pause_indices = np.where(time_diffs > 0.1)[0]  # Gaps larger than 100ms
        
        # Combine both types of changes
        all_changes = np.unique(np.concatenate(([0], semitone_changes, pause_indices, [len(filtered_times)-1])))
        all_changes.sort()
        
        # Step 4: Create note events from segments
        note_events = []
        
        for i in range(len(all_changes) - 1):
            start_idx = all_changes[i]
            end_idx = all_changes[i + 1]
            
            # Skip if indices are the same
            if start_idx == end_idx:
                continue
                
            segment_times = filtered_times[start_idx:end_idx+1]
            segment_freqs = filtered_freqs[start_idx:end_idx+1]
            segment_conf = filtered_conf[start_idx:end_idx+1]
            
            # Calculate duration
            duration = segment_times[-1] - segment_times[0]
            
            # Skip very short notes (but use a lower threshold)
            if duration < min_duration:
                continue
            
            # Calculate median frequency
            median_freq = np.median(segment_freqs)
            
            # Calculate semitone relative to tonic
            cents_from_tonic = 1200 * np.log2(median_freq / max(1.0, self.tonic['frequency']))
            semitone = round(cents_from_tonic / 100)
            
            # Calculate deviation from equal temperament
            deviation = cents_from_tonic - (semitone * 100)
            
            # Detect if note has significant pitch movement (gamaka)
            freq_std = np.std(segment_freqs)
            has_gamaka = freq_std > (median_freq * 0.03)  # Lower threshold
            
            # Store note event
            note_events.append({
                'start_time': segment_times[0],
                'end_time': segment_times[-1],
                'duration': duration,
                'frequency': median_freq,
                'semitone': semitone,
                'deviation': deviation,
                'confidence': np.mean(segment_conf),
                'has_gamaka': has_gamaka,
                'gamaka_intensity': freq_std / median_freq if has_gamaka else 0
            })
        
        # Add a fallback: if we still don't have notes but we have valid frequencies,
        # create at least one note from the highest confidence segment
        if len(note_events) == 0 and len(valid_indices) > 0:
            best_idx = np.argmax(filtered_conf)
            note_freq = filtered_freqs[best_idx]
            cents_from_tonic = 1200 * np.log2(note_freq / max(1.0, self.tonic['frequency']))
            semitone = round(cents_from_tonic / 100)
            
            note_events = [{
                'start_time': filtered_times[0],
                'end_time': filtered_times[-1],
                'duration': filtered_times[-1] - filtered_times[0],
                'frequency': note_freq,
                'semitone': semitone,
                'deviation': 0,
                'confidence': filtered_conf[best_idx],
                'has_gamaka': False,
                'gamaka_intensity': 0
            }]
            print("Created fallback note event")
        
        # Store the extracted note events
        self.note_events = note_events
        
        print(f"Extracted {len(note_events)} note events")
            # Add this improved filtering after note extraction
        filtered_notes = []
        for note in note_events:
            # Skip notes with very low confidence
            if note['confidence'] < 0.3:
                continue
                
            # Convert to scale degree relative to tonic
            cents_from_tonic = 1200 * np.log2(note['frequency'] / max(1.0, self.tonic['frequency']))
            raw_semitone = cents_from_tonic / 100
            
            # Quantize to nearest semitone with special handling for common Carnatic positions
            # In Carnatic music, some notes appear at specific microtonal positions
            if abs(raw_semitone - round(raw_semitone)) > 0.2:
                # Significant deviation from equal temperament
                # Check if it's a known microtonal position in Carnatic music
                known_microtones = {0.33: 1, 0.66: 1, 1.25: 1, 1.75: 2}
                for micro, target in known_microtones.items():
                    if abs(raw_semitone % 1 - micro) < 0.1:
                        # Map to the corresponding semitone
                        note['semitone'] = int(raw_semitone) + target
                        break
                else:
                    # Not a known microtone, use standard rounding
                    note['semitone'] = round(raw_semitone)
            else:
                # Standard semitone
                note['semitone'] = round(raw_semitone)
                
            filtered_notes.append(note)
        
        # Update note_events with filtered version
        self.note_events = filtered_notes
        return note_events
    
    def extract_phrases(self, min_notes=2, max_interval=2.0):  # Increased interval
        """
        Extract musical phrases from note events.
        
        Parameters:
        - min_notes: Minimum number of notes in a phrase
        - max_interval: Maximum time interval between notes in a phrase (in seconds)
        
        Returns:
        - List of phrases, each containing a list of note events
        """
        if self.note_events is None:
            print("No note events available. Please run extract_note_events() first.")
            return None
        
        phrases = []
        
        # Handle special case: very few notes
        if len(self.note_events) <= 3:
            # If we have at least one note, use it as a minimal phrase
            if len(self.note_events) > 0:
                phrases = [self.note_events]
                print(f"Created one minimal phrase from {len(self.note_events)} notes")
            return phrases
        
        # For larger files, try to create multiple phrases
        if len(self.note_events) > 20 and self.duration > 30:
            # Split into multiple phrases based on note timestamps
            note_times = [note['start_time'] for note in self.note_events]
            
            # Find pauses that are significantly longer than the average
            intervals = np.diff(note_times)
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            pause_threshold = max(1.5, avg_interval + 2 * std_interval)
            
            # Find significant pauses
            pause_indices = np.where(intervals > pause_threshold)[0]
            
            # Use pauses to segment phrases
            if len(pause_indices) > 0:
                start_idx = 0
                for pause_idx in pause_indices:
                    end_idx = pause_idx + 1  # Note after the pause
                    if end_idx - start_idx >= min_notes:
                        phrases.append(self.note_events[start_idx:end_idx])
                    start_idx = end_idx
                
                # Add the last phrase
                if len(self.note_events) - start_idx >= min_notes:
                    phrases.append(self.note_events[start_idx:])
                
                if phrases:
                    print(f"Created {len(phrases)} phrases using pause detection")
                    return phrases
        
        # Fall back to traditional grouping if pause detection didn't work
        current_phrase = []
        
        for i, note in enumerate(self.note_events):
            # If current_phrase is empty, add the note
            if not current_phrase:
                current_phrase = [note]
                continue
            
            # Check if this note is within the time interval of the previous note
            prev_note = current_phrase[-1]
            if note['start_time'] - prev_note['end_time'] <= max_interval:
                # Note is part of the current phrase
                current_phrase.append(note)
            else:
                # Note starts a new phrase
                if len(current_phrase) >= min_notes:
                    phrases.append(current_phrase)
                current_phrase = [note]
        
        # Add the last phrase if it meets the minimum notes requirement
        if len(current_phrase) >= min_notes:
            phrases.append(current_phrase)
        elif len(current_phrase) > 0 and len(phrases) == 0:
            # If we have no phrases yet, use whatever we have
            phrases.append(current_phrase)
        
        # If we still have no phrases but have notes, create at least one phrase
        if not phrases and self.note_events:
            # Divide the notes into roughly equal phrases
            if len(self.note_events) > 5:
                chunk_size = min(16, max(3, len(self.note_events) // 3))
                for i in range(0, len(self.note_events), chunk_size):
                    chunk = self.note_events[i:i+chunk_size]
                    if len(chunk) >= min_notes:
                        phrases.append(chunk)
            
            # If still no phrases, just use all notes as one phrase
            if not phrases:
                phrases = [self.note_events]
        
        print(f"Extracted {len(phrases)} phrases")
        
        return phrases
    
    def identify_raga(self, note_events=None, top_n=3):
        """
        Attempt to identify the raga based on note distribution and patterns.
        
        Parameters:
        - note_events: Note events to analyze, or None to use stored events
        - top_n: Number of top matches to return
        
        Returns:
        - List of (raga_id, confidence) tuples for top matches
        """
        if note_events is None:
            if self.note_events is None:
                print("No note events available. Please run extract_note_events() first.")
                return None
            note_events = self.note_events
        
        if not self.ragas:
            print("No raga information available.")
            return None
        
        # Extract semitones used in the performance
        semitones = [note['semitone'] % 12 for note in note_events]
        semitone_counts = Counter(semitones)
        
        # Calculate total notes for percentage
        total_notes = sum(semitone_counts.values())
        if total_notes == 0:
            return None
        
        # Calculate note distribution (as percentage)
        note_distribution = {note: count / total_notes for note, count in semitone_counts.items()}
        
        # Calculate match scores for each raga
        scores = {}
        
        for raga_id, raga in self.ragas.items():
            # Convert arohan/avarohan to semitones (modulo 12)
            raga_notes = set()
            if 'arohan' in raga:
                raga_notes.update([note % 12 for note in raga['arohan']])
            if 'avarohan' in raga:
                raga_notes.update([note % 12 for note in raga['avarohan']])
            
            # Skip if raga has no defined notes
            if not raga_notes:
                continue
            
            # Calculate percentage of notes that belong to this raga
            notes_in_raga = sum(note_distribution.get(note, 0) for note in raga_notes)
            
            # Calculate percentage of raga notes present in the performance
            coverage = len(set(semitones).intersection(raga_notes)) / len(raga_notes)
            
            # Weight the scores (adjust these weights as needed)
            score = (notes_in_raga * 0.7) + (coverage * 0.3)
            
            scores[raga_id] = score
        
        # Get top N matches
        top_matches = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Print results
        print("Top raga matches:")
        for raga_id, score in top_matches:
            raga_name = self.ragas[raga_id].get('name', raga_id)
            print(f"  {raga_name}: {score:.2f}")
        
        return top_matches
    
    def extract_characteristic_patterns(self, phrases, min_occurrences=2):
        """
        Extract characteristic melodic patterns from phrases.
        
        Parameters:
        - phrases: List of phrases to analyze
        - min_occurrences: Minimum occurrences for a pattern to be considered characteristic
        
        Returns:
        - List of characteristic patterns with occurrence counts
        """
        if not phrases:
            print("No phrases provided for pattern extraction.")
            return None
        
        # Convert phrases to semitone sequences
        semitone_phrases = []
        for phrase in phrases:
            semitones = [note['semitone'] % 12 for note in phrase]
            semitone_phrases.append(tuple(semitones))
        
        # Find n-grams (subsequences) in the semitone phrases
        patterns = defaultdict(int)
        
        for phrase in semitone_phrases:
            # Consider n-grams of various lengths
            for n in range(3, min(8, len(phrase) + 1)):
                for i in range(len(phrase) - n + 1):
                    pattern = phrase[i:i+n]
                    patterns[pattern] += 1
        
        # Filter patterns by minimum occurrences
        characteristic_patterns = [(pattern, count) for pattern, count in patterns.items() 
                                if count >= min_occurrences]
        
        # Sort by occurrence count (descending)
        characteristic_patterns.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Extracted {len(characteristic_patterns)} characteristic patterns")
        
        return characteristic_patterns
    
    def detect_gamakas(self, plot=False):
        """
        Detect and classify gamakas (ornamentations) in the performance.
        
        Parameters:
        - plot: Whether to plot gamaka examples
        
        Returns:
        - Dictionary with gamaka statistics
        """
        if self.note_events is None:
            print("No note events available. Please run extract_note_events() first.")
            return None
        
        # Count notes with gamakas
        gamaka_notes = [note for note in self.note_events if note['has_gamaka']]
        
        # Calculate percentage of notes with gamakas
        gamaka_percentage = len(gamaka_notes) / len(self.note_events) if self.note_events else 0
        
        # Group gamakas by semitone
        gamakas_by_semitone = defaultdict(list)
        for note in gamaka_notes:
            semitone = note['semitone'] % 12
            gamakas_by_semitone[semitone].append(note)
        
        # Calculate average gamaka intensity by semitone
        gamaka_intensity_by_semitone = {}
        for semitone, notes in gamakas_by_semitone.items():
            avg_intensity = np.mean([note['gamaka_intensity'] for note in notes])
            gamaka_intensity_by_semitone[semitone] = avg_intensity
        
        # Print results
        print(f"Detected {len(gamaka_notes)} notes with gamakas ({gamaka_percentage:.1%} of all notes)")
        print("Gamaka distribution by semitone:")
        for semitone, notes in sorted(gamakas_by_semitone.items()):
            note_name = librosa.midi_to_note(semitone + 60)[:-1]  # Remove octave number
            print(f"  {note_name}: {len(notes)} gamakas, average intensity: {gamaka_intensity_by_semitone[semitone]:.3f}")
        
        # Plot examples if requested
        if plot and gamaka_notes:
            # Plot a few example gamakas
            plt.figure(figsize=(15, 10))
            
            # Take up to 6 examples with different intensities
            examples = sorted(gamaka_notes, key=lambda x: x['gamaka_intensity'], reverse=True)
            examples = examples[:min(6, len(examples))]
            
            for i, note in enumerate(examples):
                # Find the audio segment for this note
                start_sample = int(note['start_time'] * self.sample_rate)
                end_sample = int(note['end_time'] * self.sample_rate)
                
                if start_sample >= len(self.audio) or end_sample > len(self.audio):
                    continue
                
                # Extract pitch data for this segment
                segment_times = np.arange(start_sample, end_sample) / self.sample_rate
                segment_times = segment_times - note['start_time']  # Make relative to note start
                
                # Find pitch indices for this segment
                t_indices = np.where((self.pitch_data['times'] >= note['start_time']) & 
                                    (self.pitch_data['times'] <= note['end_time']))[0]
                
                if len(t_indices) == 0:
                    continue
                
                segment_pitch = self.pitch_data['frequencies'][t_indices]
                segment_pitch_times = self.pitch_data['times'][t_indices] - note['start_time']
                
                # Plot
                plt.subplot(3, 2, i+1)
                plt.plot(segment_pitch_times, segment_pitch)
                plt.axhline(y=note['frequency'], color='r', linestyle='--')
                plt.title(f"Gamaka Example {i+1}: Intensity={note['gamaka_intensity']:.3f}")
                plt.xlabel("Time (s)")
                plt.ylabel("Frequency (Hz)")
                plt.grid(True)
            
            plt.tight_layout()
            plt.show()
        
        return {
            'gamaka_notes': len(gamaka_notes),
            'total_notes': len(self.note_events),
            'gamaka_percentage': gamaka_percentage,
            'gamakas_by_semitone': {k: len(v) for k, v in gamakas_by_semitone.items()},
            'gamaka_intensity_by_semitone': gamaka_intensity_by_semitone
        }
    
    def save_analysis(self, filename='outputs/audio_analysis.pkl'):
        """
        Save the analysis results to a file.
        
        Parameters:
        - filename: Output filename
        """
        if not hasattr(self, 'audio'):
            print("No analysis data to save.")
            return
        
        # Create data to save
        data = {
            'tonic': self.tonic,
            'note_events': self.note_events,
            'duration': self.duration
        }
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save to file
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Analysis saved to {filename}")
    
    def load_analysis(self, filename='outputs/audio_analysis.pkl'):
        """
        Load analysis results from a file.
        
        Parameters:
        - filename: Input filename
        
        Returns:
        - True if successful, False otherwise
        """
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            self.tonic = data['tonic']
            self.note_events = data['note_events']
            self.duration = data.get('duration', 0)
            
            print(f"Analysis loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading analysis: {e}")
            return False
    
    def export_patterns_to_json(self, patterns, filename='outputs/raga_patterns.json'):
        """
        Export extracted patterns to a JSON file for use in the generator.
        
        Parameters:
        - patterns: List of patterns to export
        - filename: Output filename
        
        Returns:
        - True if successful, False otherwise
        """
        if not patterns:
            print("No patterns to export.")
            return False
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Get raga ID if available
        raga_id = None
        raga_name = "Unknown"
        if hasattr(self, 'identified_raga') and self.identified_raga:
            raga_id = self.identified_raga[0][0]
            raga_name = self.ragas.get(raga_id, {}).get('name', raga_id)
        
        # Prepare data for export
        export_data = {
            'raga_id': raga_id,
            'raga_name': raga_name,
            'tonic': self.tonic['note'] if self.tonic else None,
            'patterns': [
                {
                    'notes': list(pattern[0]),
                    'count': pattern[1]
                }
                for pattern in patterns
            ]
        }
        
        # Save to JSON
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"Patterns exported to {filename}")
            return True
        except Exception as e:
            print(f"Error exporting patterns: {e}")
            return False


def analyze_audio_file(file_path, output_dir='outputs', plot=False):
    """
    Perform a complete analysis of an audio file and export the results.
    
    Parameters:
    - file_path: Path to the audio file
    - output_dir: Directory for output files
    - plot: Whether to display plots during analysis
    
    Returns:
    - Dictionary of analysis results
    """
    # Create analyzer
    analyzer = CarnaticAudioAnalyzer()
    
    # Load audio
    if not analyzer.load_audio(file_path):
        return None
    
    # Extract pitch
    analyzer.extract_pitch()
    
    # Detect tonic
    analyzer.detect_tonic(plot=plot)
    
    # Extract note events
    note_events = analyzer.extract_note_events()
    
    # Extract phrases
    phrases = analyzer.extract_phrases()
    
    # Identify raga
    identified_raga = analyzer.identify_raga()
    analyzer.identified_raga = identified_raga
    
    # Extract characteristic patterns
    patterns = analyzer.extract_characteristic_patterns(phrases)
    
    # Detect gamakas
    gamaka_stats = analyzer.detect_gamakas(plot=plot)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get filename without extension
    filename = os.path.splitext(os.path.basename(file_path))[0]
    
    # Save analysis results
    analyzer.save_analysis(os.path.join(output_dir, f"{filename}_analysis.pkl"))
    
    # Export patterns
    if patterns:
        analyzer.export_patterns_to_json(patterns, os.path.join(output_dir, f"{filename}_patterns.json"))
    
    # Return results summary
    return {
        'file': file_path,
        'duration': analyzer.duration,
        'tonic': analyzer.tonic['note'] if analyzer.tonic else None,
        'notes_detected': len(analyzer.note_events) if analyzer.note_events else 0,
        'phrases_detected': len(phrases) if phrases else 0,
        'patterns_extracted': len(patterns) if patterns else 0,
        'identified_raga': identified_raga[0][0] if identified_raga else None,
        'raga_confidence': identified_raga[0][1] if identified_raga else 0,
        'gamaka_percentage': gamaka_stats['gamaka_percentage'] if gamaka_stats else 0
    }


# Example usage
if __name__ == "__main__":
    # Check if file path is provided
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        results = analyze_audio_file(file_path, plot=True)
        
        if results:
            print("\nAnalysis Results Summary:")
            for key, value in results.items():
                print(f"  {key}: {value}")
    else:
        print("Please provide an audio file path to analyze.")
        print("Usage: python audio_analyzer.py <audio_file_path>")