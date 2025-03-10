#!/usr/bin/env python3
"""
Audio Preprocessor for Carnatic Music Analysis
--------------------------------------------
Preprocesses audio files to optimize them for Carnatic music analysis.
Functions include tonic normalization, noise reduction, segmentation,
and extraction of relevant sections from longer performances.
"""

import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from librosa.segment import recurrence_matrix
from sklearn.cluster import KMeans

class CarnaticAudioPreprocessor:
    """Preprocesses audio files for Carnatic music analysis."""
    
    def __init__(self, sample_rate=22050):
        """
        Initialize the preprocessor.
        
        Parameters:
        - sample_rate: Sample rate to use for processing
        """
        self.sample_rate = sample_rate
        self.audio = None
        self.duration = 0
        
    def load_audio(self, file_path):
        """
        Load an audio file for preprocessing.
        
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
            self.file_path = file_path
            
            print(f"Loaded audio file: {file_path}")
            print(f"Duration: {self.duration:.2f} seconds")
            
            return True
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return False
    
    def normalize_volume(self, target_db=-20):
        """
        Normalize audio volume to a target dB level.
        
        Parameters:
        - target_db: Target dB level
        
        Returns:
        - Normalized audio
        """
        if self.audio is None:
            print("No audio loaded. Call load_audio() first.")
            return None
        
        # Calculate current RMS energy
        current_rms = np.sqrt(np.mean(self.audio**2))
        current_db = 20 * np.log10(current_rms) if current_rms > 0 else -100
        
        # Calculate gain needed to reach target dB
        gain = 10**((target_db - current_db) / 20)
        
        # Apply gain
        normalized_audio = self.audio * gain
        
        # Clip to avoid distortion
        normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
        
        self.audio = normalized_audio
        print(f"Normalized audio from {current_db:.1f} dB to {target_db:.1f} dB")
        
        return normalized_audio
    
    def remove_noise(self, frame_length=2048, hop_length=512, threshold=0.05):
        """
        Remove background noise using spectral gating.
        
        Parameters:
        - frame_length: Frame length for STFT
        - hop_length: Hop length for STFT
        - threshold: Threshold for noise reduction
        
        Returns:
        - Denoised audio
        """
        if self.audio is None:
            print("No audio loaded. Call load_audio() first.")
            return None
        
        # Calculate STFT
        stft = librosa.stft(self.audio, n_fft=frame_length, hop_length=hop_length)
        
        # Convert to magnitude spectrogram
        magnitude = np.abs(stft)
        
        # Estimate noise profile from the quietest 2% of frames
        frame_energies = np.sum(magnitude, axis=0)
        sorted_indices = np.argsort(frame_energies)
        noise_length = int(0.02 * len(sorted_indices))
        noise_indices = sorted_indices[:noise_length]
        
        # Calculate noise profile
        noise_profile = np.mean(magnitude[:, noise_indices], axis=1, keepdims=True)
        
        # Apply spectral gate with threshold
        gain_mask = (magnitude - threshold * noise_profile) / magnitude
        gain_mask = np.maximum(0, gain_mask)
        
        # Apply mask
        stft_denoised = stft * gain_mask
        
        # Inverse STFT
        audio_denoised = librosa.istft(stft_denoised, hop_length=hop_length, length=len(self.audio))
        
        self.audio = audio_denoised
        print(f"Applied noise reduction with threshold {threshold:.2f}")
        
        return audio_denoised
    
    def remove_silence(self, top_db=30, frame_length=2048, hop_length=512):
        """
        Remove silent sections from the audio.
        
        Parameters:
        - top_db: Silence threshold in dB
        - frame_length: Frame length for silence detection
        - hop_length: Hop length for silence detection
        
        Returns:
        - Audio with silent sections removed
        """
        if self.audio is None:
            print("No audio loaded. Call load_audio() first.")
            return None
        
        # Detect non-silent sections
        intervals = librosa.effects.split(self.audio, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
        
        # Keep track of how much audio is removed
        total_duration = len(self.audio) / self.sample_rate
        silence_duration = total_duration - sum(interval[1] - interval[0] for interval in intervals) / self.sample_rate
        
        if len(intervals) == 0:
            print("No non-silent sections found.")
            return self.audio
        
        # Concatenate non-silent segments
        audio_without_silence = np.concatenate([self.audio[start:end] for start, end in intervals])
        
        self.audio = audio_without_silence
        self.duration = len(self.audio) / self.sample_rate
        
        print(f"Removed {silence_duration:.2f} seconds of silence ({silence_duration/total_duration:.1%} of audio)")
        
        return audio_without_silence
    
    def extract_tonic(self, plot=False):
        """
        Detect and extract the tonic pitch from the audio.
        
        Parameters:
        - plot: Whether to plot pitch histogram
        
        Returns:
        - Estimated tonic frequency in Hz
        """
        if self.audio is None:
            print("No audio loaded. Call load_audio() first.")
            return None
        
        # Extract pitch using pYIN algorithm
        f0, voiced_flag, voiced_probs = librosa.pyin(
            self.audio, 
            fmin=50,
            fmax=1000,
            sr=self.sample_rate
        )
        
        # Filter out unvoiced and zero frequencies
        valid_f0 = f0[voiced_flag & (f0 > 0)]
        
        if len(valid_f0) == 0:
            print("No valid pitch detected. Cannot extract tonic.")
            return None
        
        # Convert frequencies to cents relative to C1 (to create pitch histogram)
        cents = 1200 * np.log2(valid_f0 / librosa.note_to_hz('C1'))
        
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
        # but for simplicity, we'll just use the highest peak
        tonic_cents = peak_bin_centers[0]
        tonic_freq = librosa.midi_to_hz(tonic_cents / 100 + 24)
        tonic_note = librosa.hz_to_note(tonic_freq)
        
        print(f"Detected tonic: {tonic_note} ({tonic_freq:.2f} Hz)")
        
        # Plot histogram if requested
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(bin_edges[:-1], hist_smooth)
            plt.vlines(peak_bin_centers[:5], 0, np.max(hist_smooth), color='r', linestyle='--')
            plt.title('Pitch Histogram with Tonic Candidates')
            plt.xlabel('Cents (relative to C1)')
            plt.ylabel('Count')
            plt.axvline(x=tonic_cents, color='g', linestyle='-', label=f'Tonic: {tonic_note}')
            plt.legend()
            plt.tight_layout()
            plt.show()
        
        return tonic_freq
    
    def normalize_tonic(self, target_tonic='C4', plot=False):
        """
        Normalize the audio so the tonic pitch matches the target note.
        
        Parameters:
        - target_tonic: Target tonic note (e.g., 'C4')
        - plot: Whether to plot pitch histograms
        
        Returns:
        - Tonic-normalized audio
        """
        if self.audio is None:
            print("No audio loaded. Call load_audio() first.")
            return None
        
        # Detect current tonic
        current_tonic = self.extract_tonic(plot=plot)
        
        if current_tonic is None:
            print("Could not detect tonic. Skipping tonic normalization.")
            return self.audio
        
        # Calculate target frequency
        target_freq = librosa.note_to_hz(target_tonic)
        
        # Calculate pitch shift ratio
        ratio = target_freq / current_tonic
        
        # Check if shift is significant
        if 0.98 <= ratio <= 1.02:
            print(f"Current tonic already close to {target_tonic}. Skipping normalization.")
            return self.audio
        
        # Apply pitch shift
        n_steps = 12 * np.log2(ratio)
        normalized_audio = librosa.effects.pitch_shift(self.audio, sr=self.sample_rate, n_steps=n_steps)
        
        self.audio = normalized_audio
        print(f"Normalized tonic from {librosa.hz_to_note(current_tonic)} to {target_tonic} ({n_steps:.2f} semitones)")
        
        # Verify the new tonic if plot is True
        if plot:
            self.extract_tonic(plot=True)
        
        return normalized_audio
    
    def segment_audio(self, segment_length=60, min_segment_length=10, with_overlap=True):
        """
        Segment the audio into musically meaningful chunks by detecting section boundaries.
        
        Parameters:
        - segment_length: Target length of segments in seconds
        - min_segment_length: Minimum segment length in seconds
        - with_overlap: Whether to allow overlapping segments
        
        Returns:
        - List of audio segments
        """
        if self.audio is None:
            print("No audio loaded. Call load_audio() first.")
            return None
        
        # If audio is shorter than target segment length, return as is
        if self.duration <= segment_length:
            print(f"Audio is shorter than target segment length. Returning as single segment.")
            return [self.audio]
        
        # Extract MFCCs for structural analysis
        mfccs = librosa.feature.mfcc(y=self.audio, sr=self.sample_rate, n_mfcc=13)
        
        # Calculate recurrence matrix
        rec_matrix = recurrence_matrix(mfccs, mode='affinity', width=3)
        
        # Find segment boundaries using spectral clustering
        n_segments = max(2, int(np.ceil(self.duration / segment_length)))
        n_segments = min(10, n_segments)  # Cap at 10 segments to avoid over-segmentation
        
        # Convert recurrence matrix to feature vectors for clustering
        try:
            # Newer librosa versions
            embedding = librosa.segment.agglomerative(rec_matrix, n_segments)
            boundaries = librosa.segment.subsegment(rec_matrix, n_segments, embedding=embedding)
        except TypeError:
            # Older librosa versions
            boundaries = librosa.segment.agglomerative(rec_matrix, n_segments)
        
        boundary_times = librosa.frames_to_time(boundaries, sr=self.sample_rate)
        
        # Add start and end times
        boundary_times = np.concatenate([[0], boundary_times, [self.duration]])
        
        # Create segments
        segments = []
        for i in range(len(boundary_times) - 1):
            start_time = boundary_times[i]
            end_time = boundary_times[i+1]
            segment_duration = end_time - start_time
            
            # Skip if segment is too short
            if segment_duration < min_segment_length:
                # Try to merge with adjacent segment if possible
                if i < len(boundary_times) - 2:
                    # Merge with next segment
                    end_time = boundary_times[i+2]
                    segment_duration = end_time - start_time
                    i += 1  # Skip the next boundary in the loop
                elif i > 0:
                    # Merge with previous segment
                    continue  # Skip this segment as it was merged with previous
                else:
                    # Can't merge, use as is if it's not extremely short
                    if segment_duration < 3:  # Skip extremely short segments (< 3 seconds)
                        continue
            
            # Calculate sample indices
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            # Extract segment
            segment = self.audio[start_sample:end_sample]
            segments.append((segment, start_time, end_time))
            
            print(f"Segment {i+1}: {start_time:.2f}s - {end_time:.2f}s ({segment_duration:.2f}s)")
        
        # If we have too few segments, try a different approach
        if len(segments) < 2 and self.duration > segment_length * 2:
            # Divide into roughly equal parts
            n_equal_segments = max(2, int(self.duration / segment_length))
            n_equal_segments = min(6, n_equal_segments)  # Cap at 6 segments
            equal_segments = []
            
            segment_duration = self.duration / n_equal_segments
            for i in range(n_equal_segments):
                start_time = i * segment_duration
                end_time = (i + 1) * segment_duration
                
                # Calculate sample indices
                start_sample = int(start_time * self.sample_rate)
                end_sample = int(end_time * self.sample_rate)
                
                # Extract segment
                segment = self.audio[start_sample:end_sample]
                equal_segments.append((segment, start_time, end_time))
                
                print(f"Equal Segment {i+1}: {start_time:.2f}s - {end_time:.2f}s ({segment_duration:.2f}s)")
            
            segments = equal_segments
        
        # Add overlapping segments if requested and if we have larger segments
        if with_overlap and self.duration > 2 * segment_length and len(segments) > 0:
            # Determine a good overlap length - about half the average segment size but at least 10 seconds
            avg_segment_duration = sum(end - start for _, start, end in segments) / len(segments)
            overlap_length = max(min_segment_length, avg_segment_duration / 2)
            
            # Only add overlaps between longer segments
            for i in range(len(segments) - 1):
                _, start1, end1 = segments[i]
                _, start2, _ = segments[i+1]
                
                # Only add overlap if segments are sufficiently long
                if end1 - start1 >= min_segment_length and start2 - start1 >= min_segment_length:
                    # Create overlap centered between segments
                    overlap_center = (end1 + start2) / 2
                    overlap_start = max(start1, overlap_center - overlap_length/2)
                    overlap_end = min(start2, overlap_center + overlap_length/2)
                    
                    if overlap_end - overlap_start >= min_segment_length:
                        # Calculate sample indices
                        start_sample = int(overlap_start * self.sample_rate)
                        end_sample = int(overlap_end * self.sample_rate)
                        
                        # Extract segment
                        segment = self.audio[start_sample:end_sample]
                        segments.append((segment, overlap_start, overlap_end, "overlap"))
                        
                        print(f"Overlap Segment {i+1}: {overlap_start:.2f}s - {overlap_end:.2f}s ({overlap_end-overlap_start:.2f}s)")
        
        # Sort all segments by start time
        segments.sort(key=lambda x: x[1])
        
        # Return audio segments only
        return [segment[0] for segment in segments]
    
    def extract_alap(self, plot=False):
        """
        Extract the alap (melodic improvisation) section from a Carnatic performance.
        
        Parameters:
        - plot: Whether to plot the analysis
        
        Returns:
        - Extracted alap section
        """
        if self.audio is None:
            print("No audio loaded. Call load_audio() first.")
            return None
        
        # Alap sections typically have:
        # 1. Lower percussive content
        # 2. More stable/slower tempo
        # 3. Often at the beginning of performances
        
        # Calculate contrast features
        hop_length = 512
        oenv = librosa.onset.onset_strength(y=self.audio, sr=self.sample_rate, hop_length=hop_length)
        tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=self.sample_rate, hop_length=hop_length)
        
        # Estimate percussiveness
        percussive = librosa.effects.percussive(self.audio)
        percussive_energy = librosa.feature.rms(y=percussive, hop_length=hop_length)[0]
        
        # Normalize to 0-1 range
        percussive_energy = (percussive_energy - np.min(percussive_energy)) / (np.max(percussive_energy) - np.min(percussive_energy))
        
        # Calculate tempo stability (variation in onset strength)
        tempo_stability = np.std(tempogram, axis=0)
        tempo_stability = 1 - (tempo_stability - np.min(tempo_stability)) / (np.max(tempo_stability) - np.min(tempo_stability))
        
        # Combine metrics to get alap likelihood
        if len(percussive_energy) != len(tempo_stability):
            # Resample to match lengths
            if len(percussive_energy) > len(tempo_stability):
                percussive_energy = librosa.resample(percussive_energy, orig_sr=len(percussive_energy), target_sr=len(tempo_stability))
            else:
                tempo_stability = librosa.resample(tempo_stability, orig_sr=len(tempo_stability), target_sr=len(percussive_energy))
        
        alap_likelihood = (1 - percussive_energy) * tempo_stability
        
        # Apply temporal smoothing
        alap_likelihood = medfilt(alap_likelihood, kernel_size=13)
        
        # Find contiguous sections with high alap likelihood
        threshold = np.percentile(alap_likelihood, 70)  # Top 30%
        alap_frames = np.where(alap_likelihood > threshold)[0]
        
        if len(alap_frames) == 0:
            print("No clear alap section detected.")
            return None
        
        # Group into contiguous sections
        groups = []
        current_group = [alap_frames[0]]
        
        for i in range(1, len(alap_frames)):
            if alap_frames[i] == alap_frames[i-1] + 1:
                current_group.append(alap_frames[i])
            else:
                groups.append(current_group)
                current_group = [alap_frames[i]]
        
        if current_group:
            groups.append(current_group)
        
        # Find the largest contiguous section
        largest_group = max(groups, key=len)
        
        # Convert frames to time
        start_time = librosa.frames_to_time(largest_group[0], sr=self.sample_rate, hop_length=hop_length)
        end_time = librosa.frames_to_time(largest_group[-1], sr=self.sample_rate, hop_length=hop_length)
        
        # Calculate sample indices
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        
        # Extract alap section
        alap_section = self.audio[start_sample:end_sample]
        
        print(f"Extracted alap section: {start_time:.2f}s - {end_time:.2f}s ({end_time - start_time:.2f}s)")
        
        if plot:
            plt.figure(figsize=(12, 8))
            
            # Plot waveform
            plt.subplot(3, 1, 1)
            plt.plot(librosa.times_like(self.audio, sr=self.sample_rate), self.audio)
            plt.axvline(x=start_time, color='r', linestyle='--')
            plt.axvline(x=end_time, color='r', linestyle='--')
            plt.title('Waveform with Detected Alap Section')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            
            # Plot percussive energy
            plt.subplot(3, 1, 2)
            plt.plot(librosa.frames_to_time(np.arange(len(percussive_energy)), sr=self.sample_rate, hop_length=hop_length), percussive_energy)
            plt.axvline(x=start_time, color='r', linestyle='--')
            plt.axvline(x=end_time, color='r', linestyle='--')
            plt.title('Percussive Energy')
            plt.xlabel('Time (s)')
            plt.ylabel('Energy')
            
            # Plot alap likelihood
            plt.subplot(3, 1, 3)
            plt.plot(librosa.frames_to_time(np.arange(len(alap_likelihood)), sr=self.sample_rate, hop_length=hop_length), alap_likelihood)
            plt.axhline(y=threshold, color='g', linestyle='--', label='Threshold')
            plt.axvline(x=start_time, color='r', linestyle='--')
            plt.axvline(x=end_time, color='r', linestyle='--')
            plt.title('Alap Likelihood')
            plt.xlabel('Time (s)')
            plt.ylabel('Likelihood')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
        
        return alap_section
    
    def extract_main_sections(self, section_count=3):
        """
        Extract main sections from a Carnatic performance (alap, gat, jhala, etc.).
        
        Parameters:
        - section_count: Number of main sections to extract
        
        Returns:
        - List of audio sections
        """
        if self.audio is None:
            print("No audio loaded. Call load_audio() first.")
            return None
        
        # Extract MFCCs and rhythmic features
        hop_length = 512
        mfccs = librosa.feature.mfcc(y=self.audio, sr=self.sample_rate, n_mfcc=13, hop_length=hop_length)
        
        # Calculate onset strength
        oenv = librosa.onset.onset_strength(y=self.audio, sr=self.sample_rate, hop_length=hop_length)
        
        # Calculate tempogram
        tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=self.sample_rate, hop_length=hop_length)
        
        # Combine features
        combined_features = np.vstack([
            mfccs,
            np.mean(tempogram, axis=0, keepdims=True),
            np.std(tempogram, axis=0, keepdims=True),
            oenv.reshape(1, -1)
        ])
        
        # Transpose to get features per frame
        X = combined_features.T
        
        # Apply clustering to find sections
        kmeans = KMeans(n_clusters=section_count, random_state=42)
        labels = kmeans.fit_predict(X)
        
        # Find contiguous segments with the same label
        boundaries = np.where(np.diff(labels) != 0)[0]
        
        # Convert to time boundaries
        boundaries_time = librosa.frames_to_time(boundaries, sr=self.sample_rate, hop_length=hop_length)
        
        # Add start and end times
        boundaries_time = np.concatenate([[0], boundaries_time, [self.duration]])
        
        # Extract sections
        sections = []
        for i in range(len(boundaries_time) - 1):
            start_time = boundaries_time[i]
            end_time = boundaries_time[i+1]
            
            # Calculate sample indices
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            # Extract section
            section = self.audio[start_sample:end_sample]
            sections.append((section, start_time, end_time, f"Section {i+1}"))
            
            print(f"Section {i+1}: {start_time:.2f}s - {end_time:.2f}s ({end_time - start_time:.2f}s)")
        
        # Return sections
        return sections
    
    def bandpass_filter(self, low_freq=70, high_freq=8000):
        """
        Apply bandpass filter to focus on frequency range of interest.
        
        Parameters:
        - low_freq: Low cutoff frequency in Hz
        - high_freq: High cutoff frequency in Hz
        
        Returns:
        - Filtered audio
        """
        if self.audio is None:
            print("No audio loaded. Call load_audio() first.")
            return None
        
        # Apply bandpass filter
        filtered_audio = librosa.effects.preemphasis(self.audio)
        
        # Filter frequencies
        stft = librosa.stft(filtered_audio)
        frequencies = librosa.fft_frequencies(sr=self.sample_rate)
        
        # Create filter mask
        mask = np.ones(len(frequencies))
        mask[frequencies < low_freq] = 0
        mask[frequencies > high_freq] = 0
        
        # Apply frequency mask
        stft_filtered = stft * mask[:, np.newaxis]
        
        # Inverse STFT
        filtered_audio = librosa.istft(stft_filtered, length=len(self.audio))
        
        self.audio = filtered_audio
        print(f"Applied bandpass filter: {low_freq}Hz - {high_freq}Hz")
        
        return filtered_audio
    
    def save_audio(self, output_path=None, format='wav'):
        """
        Save the processed audio to a file.
        
        Parameters:
        - output_path: Path to save the audio file, or None to use original name with suffix
        - format: Audio format ('wav', 'mp3', 'ogg', 'flac')
        
        Returns:
        - Path to the saved file
        """
        if self.audio is None:
            print("No audio loaded. Call load_audio() first.")
            return None
        
        # Generate output path if not provided
        if output_path is None:
            if hasattr(self, 'file_path'):
                base_dir = os.path.dirname(self.file_path)
                base_name = os.path.splitext(os.path.basename(self.file_path))[0]
                output_path = os.path.join(base_dir, f"{base_name}_processed.{format}")
            else:
                output_path = f"processed_audio_{int(time.time())}.{format}"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Normalize audio to avoid clipping
        audio_normalized = np.clip(self.audio, -1.0, 1.0)
        
        # Save the file
        sf.write(output_path, audio_normalized, self.sample_rate)
        print(f"Saved processed audio to: {output_path}")
        
        return output_path
    
    def process_for_analysis(self, normalize_tonic=True, target_tonic='C4', remove_silence=True, 
                           apply_bandpass=True, normalize_volume=True, target_db=-18):
        """
        Apply a standard processing pipeline optimized for Carnatic music analysis.
        
        Parameters:
        - normalize_tonic: Whether to normalize the tonic pitch
        - target_tonic: Target tonic note for normalization
        - remove_silence: Whether to remove silent sections
        - apply_bandpass: Whether to apply bandpass filtering
        - normalize_volume: Whether to normalize audio volume
        - target_db: Target dB level for volume normalization
        
        Returns:
        - Processed audio
        """
        if self.audio is None:
            print("No audio loaded. Call load_audio() first.")
            return None
        
        # Apply processing steps in sequence
        if normalize_volume:
            self.normalize_volume(target_db=target_db)
        
        if remove_silence:
            self.remove_silence()
        
        if apply_bandpass:
            self.bandpass_filter(low_freq=60, high_freq=10000)
        
        if normalize_tonic:
            self.normalize_tonic(target_tonic=target_tonic)
        
        return self.audio


def preprocess_file(file_path, output_dir=None, normalize_tonic=True, target_tonic='C4',
                   remove_silence=True, apply_bandpass=True, normalize_volume=True, 
                   target_db=-18, plot=False):
    """
    Preprocess a single audio file with standard settings for Carnatic music analysis.
    
    Parameters:
    - file_path: Path to the audio file
    - output_dir: Directory to save processed files (None to use same directory)
    - normalize_tonic: Whether to normalize the tonic pitch
    - target_tonic: Target tonic note for normalization
    - remove_silence: Whether to remove silent sections
    - apply_bandpass: Whether to apply bandpass filtering
    - normalize_volume: Whether to normalize audio volume
    - target_db: Target dB level for volume normalization
    - plot: Whether to display plots during processing
    
    Returns:
    - Path to the processed file
    """
    # Generate output path
    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_processed.wav")
    
    # Process the file
    preprocessor = CarnaticAudioPreprocessor()
    
    if not preprocessor.load_audio(file_path):
        return None
    
    # Extract tonic before processing for reference
    if plot:
        original_tonic = preprocessor.extract_tonic(plot=True)
    
    # Apply standard processing
    preprocessor.process_for_analysis(
        normalize_tonic=normalize_tonic,
        target_tonic=target_tonic,
        remove_silence=remove_silence,
        apply_bandpass=apply_bandpass,
        normalize_volume=normalize_volume,
        target_db=target_db
    )
    
    # Check tonic after processing
    if normalize_tonic and plot:
        preprocessor.extract_tonic(plot=True)
    
    # Save processed file
    return preprocessor.save_audio(output_path)


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess Carnatic music recordings for analysis.')
    parser.add_argument('file', help='Path to audio file')
    parser.add_argument('-o', '--output-dir', help='Output directory')
    parser.add_argument('--no-tonic-norm', action='store_false', dest='normalize_tonic', 
                      help='Disable tonic normalization')
    parser.add_argument('--target-tonic', default='C4', help='Target tonic note (default: C4)')
    parser.add_argument('--no-silence-removal', action='store_false', dest='remove_silence',
                      help='Disable silence removal')
    parser.add_argument('--no-bandpass', action='store_false', dest='apply_bandpass',
                      help='Disable bandpass filtering')
    parser.add_argument('--no-volume-norm', action='store_false', dest='normalize_volume',
                      help='Disable volume normalization')
    parser.add_argument('--target-db', type=float, default=-18, 
                      help='Target dB level for volume normalization')
    parser.add_argument('-p', '--plot', action='store_true', help='Display plots during processing')
    
    args = parser.parse_args()
    
    output_path = preprocess_file(
        args.file,
        output_dir=args.output_dir,
        normalize_tonic=args.normalize_tonic,
        target_tonic=args.target_tonic,
        remove_silence=args.remove_silence,
        apply_bandpass=args.apply_bandpass,
        normalize_volume=args.normalize_volume,
        target_db=args.target_db,
        plot=args.plot
    )
    
    if output_path:
        print(f"Preprocessing completed successfully. Saved to: {output_path}")
    else:
        print("Preprocessing failed.")