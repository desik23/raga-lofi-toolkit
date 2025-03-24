#!/usr/bin/env python3
"""
Audio Processor for Raga-Lofi Integration
----------------------------------------
Core audio processing module that integrates with harmony analysis and audiocraft
generation. Handles audio transformation, effects application, and preparation
for lofi music production based on raga analysis.
"""

import os
import numpy as np
import librosa
import soundfile as sf
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
import json
import time
from scipy.signal import butter, lfilter, lfiltic

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('audio_processor')

class AudioProcessor:
    """
    Handles audio processing for raga-based lo-fi music production.
    Integrates with harmony analysis and audiocraft bridge for complete
    audio transformation pipeline.
    """
    
    def __init__(self, 
                 sample_rate: int = 44100, 
                 channels: int = 2,
                 bit_depth: int = 16):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate: Sample rate for processing (default: 44100)
            channels: Number of audio channels (default: 2)
            bit_depth: Bit depth for output (default: 16)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.bit_depth = bit_depth
        self.audio = None
        self.harmony_data = None
        self.duration = 0
        self.file_path = None
        self.effects_chain = []
        
    def load_audio(self, file_path: str) -> bool:
        """
        Load an audio file for processing.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"Audio file not found: {file_path}")
                return False
                
            # Load the audio file
            y, sr = librosa.load(file_path, sr=self.sample_rate, mono=False)
            
            # Convert to stereo if needed
            if y.ndim == 1:
                # Mono to stereo
                y = np.vstack([y, y])
            elif y.shape[0] > 2:
                # More than 2 channels, keep just the first two
                y = y[:2]
            
            self.audio = y
            self.file_path = file_path
            self.duration = librosa.get_duration(y=y[0], sr=sr)
            
            logger.info(f"Loaded audio file: {file_path}")
            logger.info(f"Duration: {self.duration:.2f} seconds, Channels: {y.shape[0]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading audio file: {str(e)}")
            return False
            
    def load_harmony_data(self, harmony_data: Union[Dict, str]) -> bool:
        """
        Load harmony analysis data.
        
        Args:
            harmony_data: Either a dictionary with harmony data or path to JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if isinstance(harmony_data, str):
                # Load from file
                if not os.path.exists(harmony_data):
                    logger.error(f"Harmony data file not found: {harmony_data}")
                    return False
                    
                with open(harmony_data, 'r') as f:
                    self.harmony_data = json.load(f)
            else:
                # Use provided dictionary
                self.harmony_data = harmony_data
                
            logger.info("Loaded harmony analysis data")
            return True
            
        except Exception as e:
            logger.error(f"Error loading harmony data: {str(e)}")
            return False
    
    def apply_lofi_effects(self, 
                         lofi_style: str = 'classic',
                         effect_intensity: float = 0.5) -> np.ndarray:
        """
        Apply lo-fi style effects to the audio.
        
        Args:
            lofi_style: Style preset ('classic', 'tape', 'vinyl', 'ambient')
            effect_intensity: Intensity of effects (0.0-1.0)
            
        Returns:
            Processed audio array
        """
        if self.audio is None:
            logger.error("No audio loaded. Call load_audio() first.")
            return None
            
        # Store original audio for potential reset
        original_audio = self.audio.copy()
        
        try:
            # Apply effects based on style preset
            if lofi_style == 'classic':
                self._apply_classic_lofi(effect_intensity)
            elif lofi_style == 'tape':
                self._apply_tape_lofi(effect_intensity)
            elif lofi_style == 'vinyl':
                self._apply_vinyl_lofi(effect_intensity)
            elif lofi_style == 'ambient':
                self._apply_ambient_lofi(effect_intensity)
            else:
                logger.warning(f"Unknown style '{lofi_style}', defaulting to 'classic'")
                self._apply_classic_lofi(effect_intensity)
                
            # Store the effect chain for potential layering with other tracks
            self.effects_chain.append({
                'type': 'lofi_style',
                'style': lofi_style,
                'intensity': effect_intensity
            })
            
            logger.info(f"Applied {lofi_style} lo-fi effects with intensity {effect_intensity:.2f}")
            
            return self.audio
            
        except Exception as e:
            # Restore original audio in case of error
            self.audio = original_audio
            logger.error(f"Error applying lo-fi effects: {str(e)}")
            return None
    
    def _apply_classic_lofi(self, intensity: float) -> None:
        """
        Apply classic lo-fi effects (bitcrushing, lowpass filter, subtle saturation).
        
        Args:
            intensity: Effect intensity (0.0-1.0)
        """
        # Apply lowpass filter
        cutoff_freq = 7500 - (intensity * 4500)  # 3000-7500 Hz range
        self._apply_lowpass_filter(cutoff_freq)
        
        # Apply bit crushing (reduce effective bit depth)
        bit_reduction = int(intensity * 8)  # Reduce by up to 8 bits
        if bit_reduction > 0:
            self._apply_bitcrusher(bit_reduction)
        
        # Apply subtle saturation
        saturation_amount = 0.1 + (intensity * 0.4)  # 0.1-0.5 range
        self._apply_saturation(saturation_amount)
        
        # Apply subtle noise
        noise_amount = 0.001 + (intensity * 0.009)  # 0.001-0.01 range
        self._add_noise(noise_amount)
    
    def _apply_tape_lofi(self, intensity: float) -> None:
        """
        Apply tape-style lo-fi effects (tape saturation, wow & flutter, hiss).
        
        Args:
            intensity: Effect intensity (0.0-1.0)
        """
        # Apply tape warmth (mid-focused EQ with saturation)
        self._apply_tape_warmth(intensity)
        
        # Apply wow & flutter (pitch/time variations)
        wow_depth = 0.001 + (intensity * 0.004)  # 0.001-0.005 range
        flutter_rate = 0.5 + (intensity * 4.5)  # 0.5-5.0 Hz range
        self._apply_wow_flutter(wow_depth, flutter_rate)
        
        # Add tape hiss
        hiss_amount = 0.001 + (intensity * 0.009)  # 0.001-0.01 range
        self._add_tape_hiss(hiss_amount)
        
        # Apply gentle lowpass filter
        cutoff_freq = 9000 - (intensity * 4000)  # 5000-9000 Hz range
        self._apply_lowpass_filter(cutoff_freq)
    
    def _apply_vinyl_lofi(self, intensity: float) -> None:
        """
        Apply vinyl-style lo-fi effects (crackle, pops, resonance).
        
        Args:
            intensity: Effect intensity (0.0-1.0)
        """
        # Apply vinyl warmth (EQ curve with resonance)
        self._apply_vinyl_eq(intensity)
        
        # Add vinyl crackle
        crackle_amount = 0.002 + (intensity * 0.018)  # 0.002-0.02 range
        self._add_vinyl_crackle(crackle_amount)
        
        # Add occasional pops
        pop_density = intensity * 0.05  # 0-0.05 range (pops per second)
        self._add_vinyl_pops(pop_density)
        
        # Apply subtle highpass filter
        highpass_freq = 20 + (intensity * 60)  # 20-80 Hz range
        self._apply_highpass_filter(highpass_freq)
    
    def _apply_ambient_lofi(self, intensity: float) -> None:
        """
        Apply ambient lo-fi effects (reverb, delay, lowpass filter).
        
        Args:
            intensity: Effect intensity (0.0-1.0)
        """
        # Apply reverb
        reverb_amount = 0.2 + (intensity * 0.5)  # 0.2-0.7 range
        decay_time = 1.0 + (intensity * 4.0)  # 1.0-5.0 seconds
        self._apply_reverb(reverb_amount, decay_time)
        
        # Apply subtle delay
        if intensity > 0.3:
            delay_time = 0.25  # Quarter note delay
            feedback = 0.1 + (intensity * 0.3)  # 0.1-0.4 range
            self._apply_delay(delay_time, feedback)
        
        # Apply gentle lowpass filter
        cutoff_freq = 8000 - (intensity * 4000)  # 4000-8000 Hz range
        self._apply_lowpass_filter(cutoff_freq)
        
        # Add subtle noise bed
        noise_amount = 0.001 + (intensity * 0.004)  # 0.001-0.005 range
        self._add_noise(noise_amount)
    
    def _apply_lowpass_filter(self, cutoff_freq: float) -> None:
        """
        Apply lowpass filter to audio.
        
        Args:
            cutoff_freq: Cutoff frequency in Hz
        """
        # Normalize frequency
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = cutoff_freq / nyquist
        
        # Design butterworth filter
        b, a = butter(4, normal_cutoff, btype='low', analog=False)
        
        # Apply to each channel
        filtered_audio = np.zeros_like(self.audio)
        for i in range(self.audio.shape[0]):
            filtered_audio[i] = lfilter(b, a, self.audio[i])
        
        self.audio = filtered_audio
    
    def _apply_highpass_filter(self, cutoff_freq: float) -> None:
        """
        Apply highpass filter to audio.
        
        Args:
            cutoff_freq: Cutoff frequency in Hz
        """
        # Normalize frequency
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = cutoff_freq / nyquist
        
        # Design butterworth filter
        b, a = butter(2, normal_cutoff, btype='high', analog=False)
        
        # Apply to each channel
        filtered_audio = np.zeros_like(self.audio)
        for i in range(self.audio.shape[0]):
            filtered_audio[i] = lfilter(b, a, self.audio[i])
        
        self.audio = filtered_audio
    
    def _apply_bitcrusher(self, bit_reduction: int) -> None:
        """
        Apply bit crusher effect.
        
        Args:
            bit_reduction: Number of bits to reduce
        """
        if bit_reduction <= 0:
            return
            
        # Calculate the number of steps
        steps = 2 ** (self.bit_depth - bit_reduction)
        
        # Quantize the signal
        crushed_audio = np.zeros_like(self.audio)
        for i in range(self.audio.shape[0]):
            # Scale to -1 to 1 range
            max_val = np.max(np.abs(self.audio[i]))
            if max_val > 0:
                scaled = self.audio[i] / max_val
                
                # Quantize
                crushed_audio[i] = np.round(scaled * steps) / steps * max_val
            else:
                crushed_audio[i] = self.audio[i]
        
        self.audio = crushed_audio
    
    def _apply_saturation(self, amount: float) -> None:
        """
        Apply saturation effect for warmth.
        
        Args:
            amount: Saturation amount (0.0-1.0)
        """
        # Simple tanh-based saturation
        saturated_audio = np.zeros_like(self.audio)
        for i in range(self.audio.shape[0]):
            # Apply saturation with blend
            processed = np.tanh(self.audio[i] * (1 + 3 * amount))
            saturated_audio[i] = (amount * processed) + ((1 - amount) * self.audio[i])
        
        self.audio = saturated_audio
    
    def _add_noise(self, amount: float) -> None:
        """
        Add background noise.
        
        Args:
            amount: Noise amount (0.0-1.0)
        """
        # Generate noise
        noise = np.random.normal(0, amount, self.audio.shape)
        
        # Add to audio
        self.audio = self.audio + noise
    
    def _apply_tape_warmth(self, intensity: float) -> None:
        """
        Apply tape-style warmth (EQ + saturation).
        
        Args:
            intensity: Effect intensity (0.0-1.0)
        """
        # First apply EQ - boost low mids, cut highs
        # We'll simulate this with a combination of filters
        
        # Low shelf boost around 250Hz
        nyquist = 0.5 * self.sample_rate
        low_shelf_freq = 250 / nyquist
        
        # Use a simple EQ approach with multiple filters
        # Low-mid boost
        b_low, a_low = butter(2, low_shelf_freq * 2, btype='low', analog=False)
        # High cut
        b_high, a_high = butter(1, 7500 / nyquist, btype='low', analog=False)
        
        # Apply to each channel
        processed_audio = np.zeros_like(self.audio)
        for i in range(self.audio.shape[0]):
            # Apply low boost with blend
            low_boost = lfilter(b_low, a_low, self.audio[i]) * (1 + intensity * 0.5)
            # Apply high cut
            high_cut = lfilter(b_high, a_high, self.audio[i])
            
            # Combine with original
            processed_audio[i] = (low_boost * 0.4) + (high_cut * 0.6)
        
        self.audio = processed_audio
        
        # Then apply saturation
        saturation_amount = 0.1 + (intensity * 0.3)
        self._apply_saturation(saturation_amount)
    
    def _apply_wow_flutter(self, depth: float, rate: float) -> None:
        """
        Apply wow & flutter effect (pitch/time variations).
        
        Args:
            depth: Depth of effect (frequency variation)
            rate: Rate of the effect (Hz)
        """
        # Generate modulation signal
        duration_samples = self.audio.shape[1]
        t = np.arange(duration_samples) / self.sample_rate
        
        # Generate wow effect (slower)
        wow_rate = rate * 0.1  # Much slower
        wow = depth * np.sin(2 * np.pi * wow_rate * t)
        
        # Generate flutter effect (faster)
        flutter = depth * 0.5 * np.sin(2 * np.pi * rate * t)
        
        # Combine
        modulation = wow + flutter
        
        # Apply variable speed effect
        # For each channel
        processed_audio = np.zeros_like(self.audio)
        for i in range(self.audio.shape[0]):
            # Create time-varied indices
            indices = np.cumsum(1.0 + modulation)
            # Scale to the original length
            indices = indices * (duration_samples - 1) / indices[-1]
            # Interpolate
            processed_audio[i] = np.interp(
                np.arange(duration_samples),
                indices,
                self.audio[i]
            )
        
        self.audio = processed_audio
    
    def _add_tape_hiss(self, amount: float) -> None:
        """
        Add tape hiss effect.
        
        Args:
            amount: Hiss amount (0.0-1.0)
        """
        # Generate filtered noise for tape hiss
        noise = np.random.normal(0, amount, self.audio.shape)
        
        # Shape the noise to be more like tape hiss (high frequency)
        nyquist = 0.5 * self.sample_rate
        b, a = butter(2, 2000 / nyquist, btype='high', analog=False)
        
        for i in range(noise.shape[0]):
            noise[i] = lfilter(b, a, noise[i])
        
        # Add to audio
        self.audio = self.audio + noise
    
    def _apply_vinyl_eq(self, intensity: float) -> None:
        """
        Apply vinyl-style EQ curve.
        
        Args:
            intensity: Effect intensity (0.0-1.0)
        """
        # Vinyl has a characteristic EQ curve
        # - Bass boost below 100Hz
        # - Subtle midrange presence
        # - High frequency roll-off
        
        nyquist = 0.5 * self.sample_rate
        
        # Bass boost filter
        b_bass, a_bass = butter(2, 100 / nyquist, btype='low', analog=False)
        
        # Midrange filter (around 1kHz)
        b_mid, a_mid = butter(2, [500 / nyquist, 2000 / nyquist], btype='band', analog=False)
        
        # High frequency roll-off
        b_high, a_high = butter(1, 7000 / nyquist, btype='low', analog=False)
        
        # Apply to each channel
        processed_audio = np.zeros_like(self.audio)
        for i in range(self.audio.shape[0]):
            # Apply bass boost with intensity
            bass = lfilter(b_bass, a_bass, self.audio[i]) * (1 + intensity * 0.8)
            
            # Apply midrange
            mid = lfilter(b_mid, a_mid, self.audio[i]) * (1 + intensity * 0.3)
            
            # Apply high roll-off
            high = lfilter(b_high, a_high, self.audio[i])
            
            # Mix together
            processed_audio[i] = (bass * 0.4) + (mid * 0.3) + (high * 0.3)
        
        self.audio = processed_audio
    
    def _add_vinyl_crackle(self, amount: float) -> None:
        """
        Add vinyl crackle effect.
        
        Args:
            amount: Crackle amount (0.0-1.0)
        """
        # Generate noise
        noise = np.random.normal(0, amount, self.audio.shape)
        
        # Shape the noise to be more like vinyl crackle
        # Use bandpass + non-linear processing
        nyquist = 0.5 * self.sample_rate
        b, a = butter(2, [2000 / nyquist, 8000 / nyquist], btype='band', analog=False)
        
        shaped_noise = np.zeros_like(noise)
        for i in range(noise.shape[0]):
            # Filter
            shaped_noise[i] = lfilter(b, a, noise[i])
            
            # Non-linear processing to create more "crackly" texture
            shaped_noise[i] = np.sign(shaped_noise[i]) * np.power(np.abs(shaped_noise[i]), 0.5)
        
        # Add to audio
        self.audio = self.audio + shaped_noise
    
    def _add_vinyl_pops(self, density: float) -> None:
        """
        Add vinyl pop/click effects.
        
        Args:
            density: Pops per second (0.0-1.0)
        """
        if density <= 0:
            return
            
        # Calculate number of pops
        num_pops = int(self.duration * density)
        if num_pops <= 0:
            return
            
        # Generate random pop locations
        pop_positions = np.random.randint(0, self.audio.shape[1], num_pops)
        
        # Create a pops layer
        pops = np.zeros_like(self.audio)
        
        # Create each pop
        for pos in pop_positions:
            # Simple exponential decay shape
            pop_length = int(0.005 * self.sample_rate)  # 5ms
            if pos + pop_length >= self.audio.shape[1]:
                continue
                
            pop_shape = np.exp(-np.arange(pop_length) / (0.001 * self.sample_rate))
            
            # Apply to both channels with random amplitude
            amplitude = np.random.uniform(0.1, 0.5)
            for i in range(pops.shape[0]):
                pops[i, pos:pos+pop_length] += amplitude * pop_shape * (1 if np.random.random() > 0.5 else -1)
        
        # Add pops to audio
        self.audio = self.audio + pops
    
    def _apply_reverb(self, amount: float, decay_time: float) -> None:
        """
        Apply reverb effect.
        
        Args:
            amount: Wet/dry mix (0.0-1.0)
            decay_time: Reverb decay time in seconds
        """
        # Simple convolution reverb
        dry_audio = self.audio.copy()
        
        # Create impulse response (exponential decay)
        decay_samples = int(decay_time * self.sample_rate)
        impulse = np.exp(-np.arange(decay_samples) / (decay_time * self.sample_rate))
        
        # Apply convolution to each channel
        reverb_audio = np.zeros_like(self.audio)
        for i in range(self.audio.shape[0]):
            # Use scipy's fftconvolve for efficiency
            from scipy.signal import fftconvolve
            reverb_audio[i] = fftconvolve(self.audio[i], impulse, mode='full')[:self.audio.shape[1]]
            
            # Normalize
            if np.max(np.abs(reverb_audio[i])) > 0:
                reverb_audio[i] *= np.max(np.abs(self.audio[i])) / np.max(np.abs(reverb_audio[i]))
        
        # Mix dry and wet signals
        self.audio = (1 - amount) * dry_audio + amount * reverb_audio
    
    def _apply_delay(self, delay_time: float, feedback: float) -> None:
        """
        Apply delay effect.
        
        Args:
            delay_time: Delay time in seconds
            feedback: Feedback amount (0.0-1.0)
        """
        # Simple delay with feedback
        dry_audio = self.audio.copy()
        delay_samples = int(delay_time * self.sample_rate)
        
        # Check if delay is possible
        if delay_samples >= self.audio.shape[1]:
            logger.warning(f"Delay time ({delay_time}s) exceeds audio duration. Skipping delay effect.")
            return
        
        # Apply delay to each channel
        delayed_audio = np.zeros_like(self.audio)
        for i in range(self.audio.shape[0]):
            # Start with the original signal
            temp_audio = self.audio[i].copy()
            
            # Apply multiple feedback iterations
            current_feedback = feedback
            for _ in range(5):  # Limit to 5 iterations
                if current_feedback < 0.01:  # Stop if feedback becomes too small
                    break
                    
                # Create delayed version
                delayed = np.zeros_like(temp_audio)
                delayed[delay_samples:] = temp_audio[:-delay_samples] * current_feedback
                
                # Add to result
                delayed_audio[i] += delayed
                
                # Update for next iteration
                current_feedback *= feedback
            
            # Add dry signal
            delayed_audio[i] += self.audio[i]
            
            # Normalize
            if np.max(np.abs(delayed_audio[i])) > 0:
                scaling_factor = np.max(np.abs(self.audio[i])) / np.max(np.abs(delayed_audio[i]))
                delayed_audio[i] *= scaling_factor
        
        # Mix with 80/20 ratio (mostly dry)
        self.audio = 0.8 * dry_audio + 0.2 * delayed_audio
    
    def harmonize_with_raga(self, raga_id: str = None, intensity: float = 0.5) -> np.ndarray:
        """
        Apply harmony transformations based on raga characteristics.
        
        Args:
            raga_id: ID of the raga to apply or None to use harmony data
            intensity: Intensity of the effect (0.0-1.0)
            
        Returns:
            Processed audio array
        """
        # Check if harmony data or raga_id is available
        if self.harmony_data is None and raga_id is None:
            logger.error("No harmony data or raga ID provided.")
            return self.audio
            
        try:
            # If we have harmony data with chord progression, use it
            chord_progression = None
            
            if self.harmony_data:
                if 'chord_progression' in self.harmony_data:
                    chord_progression = self.harmony_data['chord_progression']
                elif 'predominant_chords' in self.harmony_data:
                    # Use predominant chords if progression not available
                    chord_progression = [{"root_name": chord, "type": ""} 
                                       for chord, _ in self.harmony_data['predominant_chords']]
            
            if chord_progression:
                logger.info(f"Applying harmony transformations based on chord progression")
                return self._apply_chord_harmony(chord_progression, intensity)
            elif raga_id:
                logger.info(f"Applying harmony transformations based on raga {raga_id}")
                return self._apply_raga_harmony(raga_id, intensity)
            else:
                logger.warning("No usable harmony data found. Returning unmodified audio.")
                return self.audio
                
        except Exception as e:
            logger.error(f"Error applying harmony transformations: {str(e)}")
            return self.audio
    
    def _apply_chord_harmony(self, chord_progression, intensity: float) -> np.ndarray:
        """
        Apply harmony effects based on chord progression.
        
        Args:
            chord_progression: List of chord dictionaries
            intensity: Effect intensity (0.0-1.0)
            
        Returns:
            Processed audio
        """
        if self.audio is None:
            logger.error("No audio loaded. Call load_audio() first.")
            return None
            
        # Preserve original audio
        original_audio = self.audio.copy()
        
        try:
            # Determine segment durations based on chord progression
            num_chords = len(chord_progression)
            if num_chords == 0:
                return self.audio
                
            # Divide audio into segments based on chord progression
            segment_samples = self.audio.shape[1] // num_chords
            
            # Process each segment with its corresponding chord
            processed_audio = np.zeros_like(self.audio)
            
            for i, chord in enumerate(chord_progression):
                start_sample = i * segment_samples
                end_sample = (i + 1) * segment_samples if i < num_chords - 1 else self.audio.shape[1]
                
                # Extract segment
                segment = self.audio[:, start_sample:end_sample]
                
                # Apply chord-specific processing
                root = chord.get('root', 0)
                chord_type = chord.get('type', 'major')
                
                # Process segment based on chord
                processed_segment = self._process_segment_with_chord(segment, root, chord_type, intensity)
                
                # Copy processed segment to output
                processed_audio[:, start_sample:end_sample] = processed_segment
            
            # Blend with original based on intensity
            self.audio = (1 - intensity * 0.5) * original_audio + (intensity * 0.5) * processed_audio
            
            # Store in effects chain
            self.effects_chain.append({
                'type': 'chord_harmony',
                'chord_progression': chord_progression,
                'intensity': intensity
            })
            
            return self.audio
            
        except Exception as e:
            # Restore original audio in case of error
            self.audio = original_audio
            logger.error(f"Error applying chord harmony: {str(e)}")
            return self.audio
    
    def _process_segment_with_chord(self, segment, root, chord_type, intensity):
        """
        Process audio segment with chord-specific effects.
        
        Args:
            segment: Audio segment to process
            root: Root note of chord (0-11)
            chord_type: Type of chord ('major', 'minor', etc.)
            intensity: Effect intensity (0.0-1.0)
            
        Returns:
            Processed segment
        """
        # Create copy of segment for processing
        processed = segment.copy()
        
        # Apply chord-specific EQ
        if chord_type == 'major':
            # Boost frequencies related to major chord tones
            processed = self._boost_chord_frequencies(processed, root, [0, 4, 7], intensity)
        elif chord_type == 'minor':
            # Boost frequencies related to minor chord tones
            processed = self._boost_chord_frequencies(processed, root, [0, 3, 7], intensity)
        elif chord_type == 'dom7':
            # Boost frequencies related to dominant 7th chord tones
            processed = self._boost_chord_frequencies(processed, root, [0, 4, 7, 10], intensity)
        else:
            # Default processing for other chord types
            processed = self._boost_chord_frequencies(processed, root, [0, 7], intensity * 0.7)
        
        return processed
    
    def _boost_chord_frequencies(self, audio, root, intervals, intensity):
        """
        Boost frequencies related to chord tones.
        
        Args:
            audio: Audio to process
            root: Root note (0-11)
            intervals: Intervals to boost
            intensity: Effect intensity (0.0-1.0)
            
        Returns:
            Processed audio
        """
        # Calculate reference frequency (A4 = 440Hz)
        a4_freq = 440.0
        a4_midi = 69
        
        # Calculate frequencies for each interval
        chord_freqs = []
        for interval in intervals:
            # Calculate MIDI note
            midi_note = (root + interval) + 60  # Middle C = 60
            
            # Convert to frequency (handle multiple octaves)
            for octave in range(-1, 3):  # Octaves -1 to 2 relative to middle C
                note = midi_note + (octave * 12)
                freq = a4_freq * 2**((note - a4_midi) / 12)
                if 20 <= freq <= 20000:  # Limit to audible range
                    chord_freqs.append(freq)
        
        # Apply subtle EQ boosts at chord frequencies
        boost_amount = 0.5 + intensity * 1.5  # 0.5 to 2.0 range
        bandwidth = 0.2 + intensity * 0.4  # 0.2 to 0.6 octaves
        
        processed = audio.copy()
        
        # Apply boosts
        for freq in chord_freqs:
            # Apply EQ boost at this frequency
            processed = self._apply_peaking_eq(processed, freq, bandwidth, boost_amount)
        
        return processed
    
    def _apply_peaking_eq(self, audio, center_freq, bandwidth, gain):
        """
        Apply peaking EQ to audio.
        
        Args:
            audio: Audio to process
            center_freq: Center frequency in Hz
            bandwidth: Bandwidth in octaves
            gain: Gain factor
            
        Returns:
            Processed audio
        """
        # Simple implementation using FFT-based filtering
        # Convert to frequency domain
        from scipy.fft import rfft, irfft
        
        processed = np.zeros_like(audio)
        
        for i in range(audio.shape[0]):
            # FFT
            spectrum = rfft(audio[i])
            
            # Apply EQ
            # Calculate frequency bins
            freq_bins = np.fft.rfftfreq(len(audio[i]), 1/self.sample_rate)
            
            # Create EQ curve
            lower_freq = center_freq / (2**(bandwidth/2))
            upper_freq = center_freq * (2**(bandwidth/2))
            
            # Simple bell curve
            eq_curve = np.ones_like(freq_bins)
            for j, freq in enumerate(freq_bins):
                if lower_freq <= freq <= upper_freq:
                    # Apply bell curve
                    rel_pos = (np.log2(freq) - np.log2(lower_freq)) / (np.log2(upper_freq) - np.log2(lower_freq))
                    # Bell shape (1 at center, tapers to 0 at edges)
                    bell_value = np.sin(rel_pos * np.pi)
                    # Scale to gain
                    eq_curve[j] = 1.0 + (gain - 1.0) * bell_value**2
            
            # Apply curve
            spectrum *= eq_curve
            
            # IFFT
            processed[i] = irfft(spectrum, len(audio[i]))
        
        return processed
    
    def _apply_raga_harmony(self, raga_id: str, intensity: float) -> np.ndarray:
        """
        Apply harmony effects based on raga characteristics.
        
        Args:
            raga_id: ID of the raga
            intensity: Effect intensity (0.0-1.0)
            
        Returns:
            Processed audio
        """
        # Placeholder for now - would need raga database
        logger.warning("Raga-specific harmony processing not yet implemented")
        
        # Eventually this would load raga data and apply specific transformations
        # based on the raga's scale, characteristic phrases, etc.
        
        return self.audio
    
    def integrate_with_audiocraft(self, prompt=None, reference_audio=None, duration=None):
        """
        Integrate with AudioCraft Bridge for AI-enhanced audio generation.
        
        Args:
            prompt: Text prompt for AudioCraft generation
            reference_audio: Optional reference audio for AudioCraft
            duration: Duration in seconds for generated audio
            
        Returns:
            Generated audio array
        """
        try:
            # Import the AudioCraft bridge
            from audiocraft_bridge import get_music_gen
            
            # If no audio is loaded but reference_audio is provided, use that
            if self.audio is None and reference_audio is not None:
                if isinstance(reference_audio, str):
                    self.load_audio(reference_audio)
                elif isinstance(reference_audio, np.ndarray):
                    self.audio = reference_audio
            
            # Determine prompt if not provided
            if prompt is None and self.harmony_data:
                # Create a prompt from harmony data
                temp_prompt = "lofi music with "
                
                # Add chord information if available
                if 'chord_progression' in self.harmony_data:
                    chord_names = []
                    for chord in self.harmony_data['chord_progression'][:3]:  # First 3 chords
                        if 'root_name' in chord and 'type' in chord:
                            chord_names.append(f"{chord['root_name']} {chord['type']}")
                        elif 'root_name' in chord:
                            chord_names.append(chord['root_name'])
                    
                    if chord_names:
                        temp_prompt += f"{', '.join(chord_names)} chord progression, "
                
                # Add mood if available
                if 'raga_info' in self.harmony_data and 'mood' in self.harmony_data['raga_info']:
                    mood = self.harmony_data['raga_info']['mood']
                    temp_prompt += f"{mood} mood, "
                
                # Finalize prompt
                temp_prompt += "ambient beats"
                prompt = temp_prompt
            
            # Determine duration if not provided
            if duration is None:
                if self.audio is not None:
                    # Use current audio duration
                    duration = self.duration
                else:
                    # Default duration
                    duration = 30.0
            
            # Get the AudioCraft bridge
            bridge = get_music_gen()
            
            if reference_audio is not None or self.audio is not None:
                # Generate with reference (current audio or provided reference)
                ref = reference_audio if reference_audio is not None else self.audio
                outputs = bridge.generate_with_reference(
                    prompts=prompt,
                    reference_audio=ref,
                    duration=duration
                )
            else:
                # Generate from prompt only
                outputs = bridge.generate(
                    prompts=prompt,
                    duration=duration
                )
            
            # Store the generated audio
            if outputs:
                # AudioCraft generates mono audio, duplicate to stereo
                generated_audio = np.repeat(outputs[0][np.newaxis, :], 2, axis=0)
                self.audio = generated_audio
                self.duration = duration
                
                logger.info(f"Generated {duration:.1f}s audio with AudioCraft")
                return generated_audio
            else:
                logger.error("AudioCraft generation failed")
                return None
                
        except ImportError:
            logger.error("AudioCraft bridge not available")
            return None
        except Exception as e:
            logger.error(f"Error in AudioCraft integration: {str(e)}")
            return None
    
    def save_audio(self, output_path=None):
        """
        Save processed audio to a file.
        
        Args:
            output_path: Path to save the audio file, or None to use original name with suffix
            
        Returns:
            Path to the saved file
        """
        if self.audio is None:
            logger.error("No audio to save. Process some audio first.")
            return None
        
        try:
            # Generate output path if not provided
            if output_path is None:
                if hasattr(self, 'file_path') and self.file_path:
                    base_dir = os.path.dirname(self.file_path)
                    base_name = os.path.splitext(os.path.basename(self.file_path))[0]
                    output_path = os.path.join(base_dir, f"{base_name}_processed.wav")
                else:
                    # Generate a timestamp-based filename
                    timestamp = int(time.time())
                    output_path = f"processed_audio_{timestamp}.wav"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Normalize audio to avoid clipping
            max_val = np.max(np.abs(self.audio))
            if max_val > 0.99:
                normalized_audio = self.audio / max_val * 0.99
            else:
                normalized_audio = self.audio
            
            # Write file
            sf.write(output_path, normalized_audio.T, self.sample_rate)
            
            logger.info(f"Saved processed audio to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving audio: {str(e)}")
            return None
    
    def export_effects_chain(self, output_path=None):
        """
        Export the effects chain to a JSON file.
        
        Args:
            output_path: Path to save the JSON file, or None to use auto-generated path
            
        Returns:
            Path to the saved file
        """
        if not self.effects_chain:
            logger.warning("No effects chain to export.")
            return None
        
        try:
            # Generate output path if not provided
            if output_path is None:
                if hasattr(self, 'file_path') and self.file_path:
                    base_dir = os.path.dirname(self.file_path)
                    base_name = os.path.splitext(os.path.basename(self.file_path))[0]
                    output_path = os.path.join(base_dir, f"{base_name}_effects_chain.json")
                else:
                    # Generate a timestamp-based filename
                    timestamp = int(time.time())
                    output_path = f"effects_chain_{timestamp}.json"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Prepare export data
            export_data = {
                'timestamp': time.time(),
                'sample_rate': self.sample_rate,
                'channels': self.channels,
                'effects_chain': self.effects_chain
            }
            
            # Write file
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported effects chain to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting effects chain: {str(e)}")
            return None


def process_audio_file(file_path, harmony_data=None, lofi_style='classic', 
                      effect_intensity=0.5, output_path=None, export_effects=True):
    """
    Process an audio file with the AudioProcessor.
    
    Args:
        file_path: Path to the audio file
        harmony_data: Path to harmony analysis JSON or dictionary
        lofi_style: Lo-fi style to apply ('classic', 'tape', 'vinyl', 'ambient')
        effect_intensity: Intensity of effects (0.0-1.0)
        output_path: Path to save the processed audio
        export_effects: Whether to export the effects chain
        
    Returns:
        Path to the processed audio file
    """
    # Create processor
    processor = AudioProcessor()
    
    # Load audio
    if not processor.load_audio(file_path):
        return None
    
    # Load harmony data if provided
    if harmony_data:
        processor.load_harmony_data(harmony_data)
    
    # Apply lo-fi effects
    processor.apply_lofi_effects(lofi_style, effect_intensity)
    
    # Apply harmony if data is available
    if harmony_data:
        processor.harmonize_with_raga(intensity=effect_intensity)
    
    # Save processed audio
    result_path = processor.save_audio(output_path)
    
    # Export effects chain if requested
    if export_effects:
        processor.export_effects_chain()
    
    return result_path


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Audio Processor for Raga-Lofi Integration')
    parser.add_argument('file', help='Path to audio file')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('--harmony', help='Path to harmony analysis JSON file')
    parser.add_argument('--style', choices=['classic', 'tape', 'vinyl', 'ambient'], 
                      default='classic', help='Lo-fi style to apply')
    parser.add_argument('--intensity', type=float, default=0.5, 
                      help='Effect intensity (0.0-1.0)')
    parser.add_argument('--no-export-effects', action='store_false', dest='export_effects',
                      help='Do not export effects chain')
    
    args = parser.parse_args()
    
    result = process_audio_file(
        args.file, 
        harmony_data=args.harmony,
        lofi_style=args.style,
        effect_intensity=args.intensity,
        output_path=args.output,
        export_effects=args.export_effects
    )
    
    if result:
        print(f"Processing completed. Output saved to: {result}")
    else:
        print("Processing failed.")