#!/usr/bin/env python3
"""
Rhythm Pattern Analyzer for Carnatic Music
-----------------------------------------
Analyzes rhythmic patterns in Carnatic music recordings,
detects talas, and extracts characteristic rhythmic phrases.
"""

import os
import numpy as np
import librosa
import librosa.display
import json
import pickle
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

class RhythmPatternAnalyzer:
    """
    A class for analyzing rhythmic patterns in Carnatic music recordings.
    """
    
    def __init__(self, sample_rate=22050):
        """
        Initialize the analyzer.
        
        Parameters:
        - sample_rate: Sample rate to use for analysis
        """
        self.sample_rate = sample_rate
        self.audio = None
        self.duration = 0
        self.onset_env = None
        self.onsets = None
        self.tempo = None
        self.beats = None
        
        # Load tala information if available
        self.talas = self._load_tala_info()
    
    def _load_tala_info(self):
        """Load tala information from a JSON file."""
        talas = {}
        try:
            with open('data/talas.json', 'r') as f:
                data = json.load(f)
                talas = data.get('talas', {})
        except (FileNotFoundError, json.JSONDecodeError):
            # Create basic tala definitions if file not found
            talas = {
                "adi": {"name": "Adi Tala", "beats": 8, "pattern": [4, 2, 2]},
                "rupaka": {"name": "Rupaka Tala", "beats": 6, "pattern": [2, 4]},
                "misra_chapu": {"name": "Misra Chapu", "beats": 7, "pattern": [3, 4]},
                "khanda_chapu": {"name": "Khanda Chapu", "beats": 5, "pattern": [2, 3]},
                "tisra_eka": {"name": "Tisra Eka", "beats": 3, "pattern": [3]},
                "chatusra_eka": {"name": "Chatusra Eka", "beats": 4, "pattern": [4]}
            }
        return talas
    
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
            self.onset_env = None
            self.onsets = None
            self.tempo = None
            self.beats = None
            
            return True
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return False
    
    def detect_onsets(self, hop_length=512):
        """
        Detect note onsets in the audio.
        
        Parameters:
        - hop_length: Hop length for onset detection
        
        Returns:
        - onset_times: Array of onset times in seconds
        """
        if self.audio is None:
            print("No audio loaded. Please call load_audio() first.")
            return None
        
        # Compute onset strength envelope
        onset_env = librosa.onset.onset_strength(
            y=self.audio, 
            sr=self.sample_rate,
            hop_length=hop_length
        )
        
        # Find onset frames
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.sample_rate,
            hop_length=hop_length,
            units='frames'
        )
        
        # Convert frames to time
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sample_rate, hop_length=hop_length)
        
        # Store results
        self.onset_env = onset_env
        self.onsets = onset_times
        
        print(f"Detected {len(onset_times)} onsets")
        
        return onset_times
    
    def estimate_tempo(self, start_bpm=60, hop_length=512):
        """
        Estimate tempo and beat positions.
        
        Parameters:
        - start_bpm: Starting tempo estimate in BPM
        - hop_length: Hop length for tempo estimation
        
        Returns:
        - tempo: Estimated tempo in BPM
        - beat_times: Array of beat times in seconds
        """
        if self.audio is None:
            print("No audio loaded. Please call load_audio() first.")
            return None, None
        
        # If onset envelope not computed yet, compute it
        if self.onset_env is None:
            self.onset_env = librosa.onset.onset_strength(
                y=self.audio, 
                sr=self.sample_rate,
                hop_length=hop_length
            )
        
        # Estimate global tempo
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=self.onset_env,
            sr=self.sample_rate,
            hop_length=hop_length,
            start_bpm=start_bpm
        )
        
        # Convert frames to time
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate, hop_length=hop_length)
        
        # Store results
        self.tempo = tempo
        self.beats = beat_times
        
        print(f"Estimated tempo: {tempo:.1f} BPM")
        print(f"Detected {len(beat_times)} beats")
        
        return tempo, beat_times
    
    def analyze_rhythm_patterns(self, plot=False):
        """
        Analyze rhythmic patterns in the audio.
        
        Parameters:
        - plot: Whether to plot rhythm analysis results
        
        Returns:
        - Dictionary with rhythm analysis results
        """
        if self.audio is None or self.onsets is None:
            print("Missing audio data or onsets. Run load_audio() and detect_onsets() first.")
            return None
        
        if self.beats is None:
            self.estimate_tempo()
        
        # Measure inter-onset intervals
        ioi = np.diff(self.onsets)
        
        # Analyze IOI distribution to find pulses
        hist, bin_edges = np.histogram(ioi, bins=50, range=(0, 1))
        pulse_candidates = []
        
        # Find peaks in the IOI histogram
        peaks, _ = find_peaks(hist, height=max(hist)/5)
        for peak in peaks:
            pulse_val = (bin_edges[peak] + bin_edges[peak+1]) / 2
            if 0.1 <= pulse_val <= 0.8:  # Reasonable range for pulses (125-600 BPM)
                pulse_candidates.append((pulse_val, hist[peak]))
        
        # Sort by count (descending)
        pulse_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Try to identify potential tala structure
        detected_tala = self._identify_tala_structure()
        
        # Identify common rhythm patterns
        rhythm_patterns = self._extract_rhythm_patterns()
        
        # Prepare results
        results = {
            'tempo': self.tempo,
            'median_pulse': np.median(ioi),
            'pulse_candidates': pulse_candidates[:3],
            'detected_tala': detected_tala,
            'rhythm_patterns': rhythm_patterns
        }
        
        # Plot results if requested
        if plot:
            self._plot_rhythm_analysis(ioi, hist, bin_edges)
        
        return results
    
    def _identify_tala_structure(self, tolerance=0.15):
        """
        Try to identify the tala structure from the beats.
        
        Parameters:
        - tolerance: Tolerance for identifying beat groupings
        
        Returns:
        - Dictionary with detected tala information
        """
        if self.beats is None or len(self.beats) < 16:
            # Need a reasonable number of beats to identify the tala
            return {'name': 'Unknown', 'confidence': 0.0, 'details': None}
        
        # Compute beat strengths based on onset strength at each beat
        beat_frames = librosa.time_to_frames(self.beats, sr=self.sample_rate)
        beat_strengths = np.array([self.onset_env[min(len(self.onset_env)-1, int(frame))] 
                                  for frame in beat_frames])
        
        # Normalize beat strengths
        beat_strengths = (beat_strengths - np.min(beat_strengths)) / (np.max(beat_strengths) - np.min(beat_strengths))
        
        # Check for recurring strong beat patterns
        potential_cycles = []
        
        # Try different cycle lengths common in Carnatic talas (3, 4, 5, 6, 7, 8, 10, 16)
        for cycle_length in [3, 4, 5, 6, 7, 8, 10, 16]:
            if len(beat_strengths) < 2 * cycle_length:
                continue
                
            # Compute autocorrelation of beat strengths
            auto_corr = np.correlate(beat_strengths, beat_strengths, mode='full')
            auto_corr = auto_corr[len(auto_corr)//2:]
            
            # Check for peak at the cycle_length
            if cycle_length < len(auto_corr) and auto_corr[cycle_length] > 0.6 * auto_corr[0]:
                score = auto_corr[cycle_length] / auto_corr[0]
                potential_cycles.append((cycle_length, score))
        
        # Find the best match
        if potential_cycles:
            potential_cycles.sort(key=lambda x: x[1], reverse=True)
            best_cycle = potential_cycles[0]
            
            # Try to identify the specific tala
            detected_tala = 'Unknown'
            best_match_score = 0
            best_tala_details = None
            
            for tala_id, tala_info in self.talas.items():
                if tala_info['beats'] == best_cycle[0]:
                    # Check beat pattern against known tala pattern
                    pattern_match_score = self._match_tala_pattern(tala_info['pattern'], beat_strengths)
                    
                    if pattern_match_score > best_match_score:
                        detected_tala = tala_info['name']
                        best_match_score = pattern_match_score
                        best_tala_details = {
                            'id': tala_id,
                            'beats': tala_info['beats'],
                            'pattern': tala_info['pattern']
                        }
            
            return {
                'name': detected_tala,
                'cycle_length': best_cycle[0],
                'confidence': best_cycle[1] * best_match_score,
                'details': best_tala_details
            }
        
        return {'name': 'Unknown', 'confidence': 0.0, 'details': None}
    
    def _match_tala_pattern(self, tala_pattern, beat_strengths):
        """
        Match the detected beat pattern against a known tala pattern.
        
        Parameters:
        - tala_pattern: List of beat counts for each anga of the tala
        - beat_strengths: Array of detected beat strengths
        
        Returns:
        - Match score (0-1)
        """
        # Create expected beat strength pattern from tala
        cycle_length = sum(tala_pattern)
        
        # Ensure we have enough data
        if len(beat_strengths) < cycle_length:
            return 0.0
        
        # Create an expected pattern with stronger first beat of each anga
        expected_pattern = np.zeros(cycle_length)
        position = 0
        for anga_length in tala_pattern:
            expected_pattern[position] = 1.0  # First beat of anga is strong
            for i in range(1, anga_length):
                expected_pattern[position + i] = 0.5  # Other beats are medium
            position += anga_length
        
        # Compute correlation between expected and actual patterns
        best_correlation = 0
        
        # Try different starting positions
        for start in range(len(beat_strengths) - cycle_length):
            current_pattern = beat_strengths[start:start+cycle_length]
            corr = np.corrcoef(expected_pattern, current_pattern)[0, 1]
            best_correlation = max(best_correlation, corr)
        
        return max(0, best_correlation)  # Ensure non-negative
    
    def _extract_rhythm_patterns(self, min_pattern_length=3, min_occurrences=2):
        """
        Extract common rhythm patterns from the onset data.
        
        Parameters:
        - min_pattern_length: Minimum number of onsets in a pattern
        - min_occurrences: Minimum occurrences for a pattern to be considered common
        
        Returns:
        - List of common rhythm patterns with counts
        """
        if self.onsets is None or len(self.onsets) < min_pattern_length * 2:
            return []
        
        # Convert onsets to relative timings
        ioi = np.diff(self.onsets)
        
        # Quantize IOIs to categories for pattern detection
        # This helps account for small timing variations
        quantized_ioi = []
        for interval in ioi:
            if interval < 0.1:
                quantized_ioi.append('VS')  # Very short
            elif interval < 0.2:
                quantized_ioi.append('S')   # Short
            elif interval < 0.4:
                quantized_ioi.append('M')   # Medium
            elif interval < 0.8:
                quantized_ioi.append('L')   # Long
            else:
                quantized_ioi.append('VL')  # Very long
        
        # Find recurring patterns
        patterns = defaultdict(int)
        
        for pattern_length in range(min_pattern_length, min(10, len(quantized_ioi) // 2)):
            for i in range(len(quantized_ioi) - pattern_length + 1):
                pattern = tuple(quantized_ioi[i:i+pattern_length])
                patterns[pattern] += 1
        
        # Filter by minimum occurrences
        common_patterns = [(pattern, count) for pattern, count in patterns.items() 
                           if count >= min_occurrences]
        
        # Sort by count (descending)
        common_patterns.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to readable format
        result = []
        for pattern, count in common_patterns[:10]:  # Top 10 patterns
            pattern_str = '-'.join(pattern)
            result.append({
                'pattern': pattern_str,
                'count': count,
                'approx_duration': sum(self._ioi_category_to_approx_time(cat) for cat in pattern)
            })
        
        return result
    
    def _ioi_category_to_approx_time(self, category):
        """Convert IOI category to approximate time value."""
        mapping = {
            'VS': 0.05,
            'S': 0.15,
            'M': 0.3,
            'L': 0.6,
            'VL': 1.0
        }
        return mapping.get(category, 0.3)
    
    def _plot_rhythm_analysis(self, ioi, hist, bin_edges):
        """
        Plot rhythm analysis results.
        
        Parameters:
        - ioi: Inter-onset intervals
        - hist: Histogram values
        - bin_edges: Bin edges for the histogram
        """
        plt.figure(figsize=(15, 10))
        
        # Plot onset envelope and beats
        plt.subplot(3, 1, 1)
        times = librosa.times_like(self.onset_env, sr=self.sample_rate)
        plt.plot(times, self.onset_env)
        if self.beats is not None:
            plt.vlines(self.beats, 0, np.max(self.onset_env), color='r', linestyle='--', alpha=0.7)
        plt.title('Onset Strength and Detected Beats')
        plt.xlabel('Time (s)')
        plt.ylabel('Strength')
        
        # Plot IOI histogram
        plt.subplot(3, 1, 2)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.bar(bin_centers, hist, width=bin_centers[1] - bin_centers[0])
        plt.title('Inter-Onset Interval Distribution')
        plt.xlabel('IOI (s)')
        plt.ylabel('Count')
        
        # Plot IOI sequence
        plt.subplot(3, 1, 3)
        plt.plot(ioi)
        plt.title('Inter-Onset Interval Sequence')
        plt.xlabel('Onset pair index')
        plt.ylabel('IOI (s)')
        
        plt.tight_layout()
        plt.show()
    
    def export_rhythm_patterns(self, filename='outputs/rhythm_patterns.json'):
        """
        Export analyzed rhythm patterns to a JSON file.
        
        Parameters:
        - filename: Output filename
        
        Returns:
        - True if successful, False otherwise
        """
        if self.onsets is None:
            print("No rhythm data to export. Run detect_onsets() first.")
            return False
        
        # Get rhythm analysis results
        results = self.analyze_rhythm_patterns()
        if not results:
            return False
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Prepare export data
        export_data = {
            'tempo': float(results['tempo']),
            'detected_tala': results['detected_tala'],
            'rhythm_patterns': results['rhythm_patterns']
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"Rhythm patterns exported to {filename}")
            return True
        except Exception as e:
            print(f"Error exporting rhythm patterns: {e}")
            return False
    
    def generate_tala_template(self, tala_id='adi', nadai=4, cycles=2):
        """
        Generate a template pattern for a specific tala, useful for comparison.
        
        Parameters:
        - tala_id: ID of the tala to generate
        - nadai: Subdivision level (e.g., 4 for chatusra)
        - cycles: Number of tala cycles to generate
        
        Returns:
        - List of expected beat times within the tala structure
        """
        if tala_id not in self.talas:
            print(f"Tala '{tala_id}' not found in available talas.")
            return None
        
        tala_info = self.talas[tala_id]
        cycle_length = tala_info['beats']
        pattern = tala_info['pattern']
        
        # Create template for one cycle
        template_cycle = []
        position = 0
        for anga_length in pattern:
            for i in range(anga_length):
                beat_type = 'strong' if i == 0 else 'normal'
                template_cycle.append({
                    'position': position,
                    'type': beat_type,
                    'anga_index': len(template_cycle) // anga_length
                })
                position += 1
        
        # Apply nadai (subdivisions)
        template_with_nadai = []
        for beat in template_cycle:
            for sub in range(nadai):
                template_with_nadai.append({
                    'position': beat['position'] * nadai + sub,
                    'type': beat['type'] if sub == 0 else 'subdivision',
                    'anga_index': beat['anga_index'],
                    'subdivision': sub
                })
        
        # Repeat for requested number of cycles
        full_template = []
        for cycle in range(cycles):
            cycle_offset = cycle * cycle_length * nadai
            for beat in template_with_nadai:
                full_template.append({
                    'position': beat['position'] + cycle_offset,
                    'type': beat['type'],
                    'anga_index': beat['anga_index'],
                    'subdivision': beat['subdivision'],
                    'cycle': cycle
                })
        
        return full_template


def analyze_rhythm(file_path, output_dir='outputs', plot=False):
    """
    Perform rhythm analysis on an audio file and export the results.
    
    Parameters:
    - file_path: Path to the audio file
    - output_dir: Directory for output files
    - plot: Whether to display plots during analysis
    
    Returns:
    - Dictionary of rhythm analysis results
    """
    # Create analyzer
    analyzer = RhythmPatternAnalyzer()
    
    # Load audio
    if not analyzer.load_audio(file_path):
        return None
    
    # Detect onsets
    analyzer.detect_onsets()
    
    # Estimate tempo
    analyzer.estimate_tempo()
    
    # Analyze rhythm patterns
    results = analyzer.analyze_rhythm_patterns(plot=plot)
    
    if not results:
        print("Failed to analyze rhythm patterns.")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get filename without extension
    filename = os.path.splitext(os.path.basename(file_path))[0]
    
    # Export results
    analyzer.export_rhythm_patterns(os.path.join(output_dir, f"{filename}_rhythm.json"))
    
    return results


# Example usage
if __name__ == "__main__":
    # Check if file path is provided
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        results = analyze_rhythm(file_path, plot=True)
        
        if results:
            print("\nRhythm Analysis Results Summary:")
            print(f"  Tempo: {results['tempo']:.1f} BPM")
            print(f"  Detected Tala: {results['detected_tala']['name']} (Confidence: {results['detected_tala']['confidence']:.2f})")
            print("\n  Common Rhythm Patterns:")
            for i, pattern in enumerate(results['rhythm_patterns'][:5]):
                print(f"    Pattern {i+1}: {pattern['pattern']} (Count: {pattern['count']})")
    else:
        print("Please provide an audio file path to analyze.")
        print("Usage: python rhythm_pattern_analyzer.py <audio_file_path>")