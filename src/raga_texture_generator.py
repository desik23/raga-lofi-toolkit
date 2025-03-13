#!/usr/bin/env python3
"""
Raga Ambient Texture Generator
-----------------------------
Utility to create custom ambient textures for ragas using
tanpura drones and audio processing.
"""

import os
import numpy as np
import argparse
import time
import random
from scipy.io import wavfile
from scipy import signal
import soundfile as sf

class RagaTextureGenerator:
    """Generate ambient textures based on raga characteristics."""
    
    def __init__(self, output_dir="generated_textures"):
        """
        Initialize the texture generator.
        
        Parameters:
        - output_dir: Directory for output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_tanpura_texture(self, tonic_freq=261.63, duration=60, sample_rate=44100, mood="peaceful"):
        """
        Generate a tanpura-like drone ambient texture.
        
        Parameters:
        - tonic_freq: Frequency of Sa (tonic note) in Hz
        - duration: Duration in seconds
        - sample_rate: Sample rate in Hz
        - mood: Texture mood ("peaceful", "devotional", "contemplative", "romantic", "melancholic")
        
        Returns:
        - Path to the generated texture file
        """
        # Create time array
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Generate base frequencies (Sa, Pa, Sa', Sa)
        sa = tonic_freq
        pa = tonic_freq * 3/2  # Perfect fifth
        sa_octave = tonic_freq * 2  # Upper octave
        
        # Create base waveforms with different amplitudes based on mood
        if mood == "peaceful":
            sa_amp = 0.5
            pa_amp = 0.3
            sa_octave_amp = 0.2
            sa_lower_amp = 0.3
        elif mood == "devotional":
            sa_amp = 0.4
            pa_amp = 0.4
            sa_octave_amp = 0.3
            sa_lower_amp = 0.3
        elif mood == "contemplative":
            sa_amp = 0.4
            pa_amp = 0.2
            sa_octave_amp = 0.2
            sa_lower_amp = 0.4
        elif mood == "romantic":
            sa_amp = 0.3
            pa_amp = 0.4
            sa_octave_amp = 0.4
            sa_lower_amp = 0.2
        elif mood == "melancholic":
            sa_amp = 0.3
            pa_amp = 0.2
            sa_octave_amp = 0.1
            sa_lower_amp = 0.5
        else:
            # Default
            sa_amp = 0.4
            pa_amp = 0.3
            sa_octave_amp = 0.2
            sa_lower_amp = 0.3
        
        # Base waveforms with slight detuning for natural sound
        detune_factor = 0.001  # 0.1% detuning
        
        sa_wave = sa_amp * np.sin(2 * np.pi * sa * t)
        sa_wave += sa_amp * 0.6 * np.sin(2 * np.pi * sa * (1 + detune_factor) * t)
        
        pa_wave = pa_amp * np.sin(2 * np.pi * pa * t)
        pa_wave += pa_amp * 0.5 * np.sin(2 * np.pi * pa * (1 - detune_factor) * t)
        
        sa_octave_wave = sa_octave_amp * np.sin(2 * np.pi * sa_octave * t)
        sa_octave_wave += sa_octave_amp * 0.4 * np.sin(2 * np.pi * sa_octave * (1 + detune_factor) * t)
        
        sa_lower_wave = sa_lower_amp * np.sin(2 * np.pi * (sa/2) * t)
        
        # Add harmonics
        for i in range(2, 5):
            sa_wave += (sa_amp / i) * np.sin(2 * np.pi * (sa * i) * t)
            pa_wave += (pa_amp / i) * np.sin(2 * np.pi * (pa * i) * t)
            sa_octave_wave += (sa_octave_amp / i) * np.sin(2 * np.pi * (sa_octave * i) * t)
            sa_lower_wave += (sa_lower_amp / i) * np.sin(2 * np.pi * ((sa/2) * i) * t)
        
        # Add pluck-like envelopes
        pluck_interval = 2.0  # Seconds between plucks
        pluck_duration = 0.5  # Duration of pluck envelope
        
        pluck_times = np.arange(0, duration, pluck_interval)
        pluck_envelope = np.zeros_like(t)
        
        for pluck_time in pluck_times:
            start_idx = int(pluck_time * sample_rate)
            end_idx = min(int((pluck_time + pluck_duration) * sample_rate), len(t))
            
            if end_idx > start_idx:
                # Create attack and decay
                attack = 0.1  # seconds
                decay = pluck_duration - attack
                
                attack_samples = int(attack * sample_rate)
                decay_samples = int(decay * sample_rate)
                
                attack_env = np.linspace(0, 1, attack_samples)
                decay_env = np.linspace(1, 0.2, decay_samples)
                
                env = np.concatenate((attack_env, decay_env))
                env_len = min(len(env), end_idx - start_idx)
                
                pluck_envelope[start_idx:start_idx + env_len] += env[:env_len]
        
        # Normalize pluck envelope to 0-1 range
        if np.max(pluck_envelope) > 0:
            pluck_envelope /= np.max(pluck_envelope)
        
        # Create slight variation in plucks for different strings
        sa_plucks = np.roll(pluck_envelope, int(random.uniform(0.1, 0.3) * sample_rate))
        pa_plucks = np.roll(pluck_envelope, int(random.uniform(0.4, 0.7) * sample_rate))
        sa_octave_plucks = np.roll(pluck_envelope, int(random.uniform(0.8, 1.2) * sample_rate))
        sa_lower_plucks = np.roll(pluck_envelope, int(random.uniform(1.3, 1.7) * sample_rate))
        
        # Apply pluck envelopes with sustain
        sa_signal = sa_wave * (0.4 + 0.6 * sa_plucks)
        pa_signal = pa_wave * (0.4 + 0.6 * pa_plucks)
        sa_octave_signal = sa_octave_wave * (0.4 + 0.6 * sa_octave_plucks)
        sa_lower_signal = sa_lower_wave * (0.4 + 0.6 * sa_lower_plucks)
        
        # Combine all signals
        combined_signal = sa_signal + pa_signal + sa_octave_signal + sa_lower_signal
        
        # Apply mild distortion for warmth (tanh soft clipping)
        distortion_amount = 1.2  # Adjust for more/less distortion
        combined_signal = np.tanh(combined_signal * distortion_amount)
        
        # Normalize
        combined_signal = combined_signal / np.max(np.abs(combined_signal))
        
        # Add gentle noise floor for lo-fi character
        noise_floor = 0.005 * np.random.randn(len(combined_signal))
        combined_signal += noise_floor
        
        # Fade in/out
        fade_samples = int(2.0 * sample_rate)  # 2-second fade
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        combined_signal[:fade_samples] *= fade_in
        combined_signal[-fade_samples:] *= fade_out
        
        # Final normalization
        combined_signal = 0.9 * combined_signal / np.max(np.abs(combined_signal))
        
        # Generate output filename
        timestamp = int(time.time())
        note_name = self._freq_to_note_name(tonic_freq)
        output_filename = f"tanpura_{note_name}_{mood}_{timestamp}.wav"
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Save as WAV
        wavfile.write(output_path, sample_rate, combined_signal.astype(np.float32))
        print(f"Generated tanpura texture: {output_path}")
        
        return output_path
    
    def process_audio_file(self, input_file, mood="peaceful", output_file=None):
        """
        Process an existing audio file to create an ambient texture.
        
        Parameters:
        - input_file: Path to the input audio file
        - mood: Texture mood ("peaceful", "devotional", "contemplative", "romantic", "melancholic")
        - output_file: Path to the output file (None for auto-generated)
        
        Returns:
        - Path to the processed file
        """
        # Check if file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file not found: {input_file}")
            return None
        
        try:
            # Load the audio file
            data, sample_rate = sf.read(input_file)
            
            # Convert to mono if needed
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            # Apply different processing based on mood
            if mood == "peaceful":
                # Gentle processing with subtle modulation
                data = self._apply_peaceful_processing(data, sample_rate)
            elif mood == "devotional":
                # Reverberant, spiritual processing
                data = self._apply_devotional_processing(data, sample_rate)
            elif mood == "contemplative":
                # Introspective, meditative processing
                data = self._apply_contemplative_processing(data, sample_rate)
            elif mood == "romantic":
                # Warm, emotive processing
                data = self._apply_romantic_processing(data, sample_rate)
            elif mood == "melancholic":
                # Somber, introspective processing
                data = self._apply_melancholic_processing(data, sample_rate)
            else:
                # Default processing
                data = self._apply_peaceful_processing(data, sample_rate)
            
            # Generate output filename if not provided
            if output_file is None:
                timestamp = int(time.time())
                filename = os.path.splitext(os.path.basename(input_file))[0]
                output_file = os.path.join(self.output_dir, f"{filename}_{mood}_{timestamp}.wav")
            
            # Save as WAV
            sf.write(output_file, data, sample_rate)
            print(f"Processed ambient texture: {output_file}")
            
            return output_file
            
        except Exception as e:
            print(f"Error processing audio file: {e}")
            return None
    
    def _apply_peaceful_processing(self, data, sample_rate):
        """Apply processing for peaceful mood."""
        # Low-pass filter for softness
        sos = signal.butter(2, 5000, 'lp', fs=sample_rate, output='sos')
        data = signal.sosfilt(sos, data)
        
        # Add gentle reverb (simplified simulation)
        reverb_time = 1.0  # seconds
        reverb_samples = int(reverb_time * sample_rate)
        decay = np.exp(-6.0 * np.arange(reverb_samples) / reverb_samples)
        
        # Create impulse response
        impulse = np.zeros(reverb_samples)
        impulse[0] = 1.0
        impulse += 0.1 * np.random.randn(reverb_samples) * decay
        impulse *= decay
        
        # Apply reverb through convolution
        data_reverb = signal.fftconvolve(data, impulse, mode='full')[:len(data)]
        
        # Mix dry/wet
        data = 0.4 * data + 0.6 * data_reverb
        
        return data
    
    def _apply_devotional_processing(self, data, sample_rate):
        """Apply processing for devotional mood."""
        # Bandpass filter centered on midrange
        sos = signal.butter(2, [400, 4000], 'bp', fs=sample_rate, output='sos')
        data = signal.sosfilt(sos, data)
        
        # Longer reverb
        reverb_time = 3.0  # seconds
        reverb_samples = int(reverb_time * sample_rate)
        decay = np.exp(-6.0 * np.arange(reverb_samples) / reverb_samples)
        
        # Create impulse response with more early reflections
        impulse = np.zeros(reverb_samples)
        impulse[0] = 1.0
        
        # Add early reflections
        for i in range(10):
            pos = int((i+1) * 0.05 * reverb_samples)
            if pos < reverb_samples:
                impulse[pos] = 0.5 * (1.0 - i/10.0)
        
        impulse *= decay
        
        # Apply reverb through convolution
        data_reverb = signal.fftconvolve(data, impulse, mode='full')[:len(data)]
        
        # Mix dry/wet (mostly wet)
        data = 0.2 * data + 0.8 * data_reverb
        
        return data
    
    def _apply_contemplative_processing(self, data, sample_rate):
        """Apply processing for contemplative mood."""
        # High-pass filter to remove rumble
        sos_hp = signal.butter(2, 100, 'hp', fs=sample_rate, output='sos')
        data = signal.sosfilt(sos_hp, data)
        
        # Low-pass filter for softness
        sos_lp = signal.butter(2, 3000, 'lp', fs=sample_rate, output='sos')
        data = signal.sosfilt(sos_lp, data)
        
        # Medium reverb
        reverb_time = 2.0  # seconds
        reverb_samples = int(reverb_time * sample_rate)
        decay = np.exp(-6.0 * np.arange(reverb_samples) / reverb_samples)
        
        # Create impulse response
        impulse = np.zeros(reverb_samples)
        impulse[0] = 1.0
        impulse += 0.05 * np.random.randn(reverb_samples) * decay
        impulse *= decay
        
        # Apply reverb through convolution
        data_reverb = signal.fftconvolve(data, impulse, mode='full')[:len(data)]
        
        # Create subtle delay (echo)
        delay_time = 0.5  # seconds
        delay_samples = int(delay_time * sample_rate)
        delay_gain = 0.3
        
        data_delay = np.zeros_like(data)
        data_delay[delay_samples:] = data[:-delay_samples] * delay_gain
        
        # Mix dry/wet with delay
        data = 0.3 * data + 0.5 * data_reverb + 0.2 * data_delay
        
        return data
    
    def _apply_romantic_processing(self, data, sample_rate):
        """Apply processing for romantic mood."""
        # Apply subtle warmth with mild distortion
        data = np.tanh(data * 1.2)
        
        # Low-pass filter for warmth
        sos = signal.butter(2, 4000, 'lp', fs=sample_rate, output='sos')
        data = signal.sosfilt(sos, data)
        
        # Add gentle chorus effect (simplified)
        depth = 0.002  # 0.2% pitch variation
        rate = 0.5  # Hz
        
        # Create modulation function
        mod_samples = int(sample_rate / rate)
        mod = depth * np.sin(2 * np.pi * np.arange(len(data)) / mod_samples)
        
        # Create chorus by resampling with variable delay
        chorus = np.zeros_like(data)
        indices = np.arange(len(data))
        indices_mod = indices - np.round(mod * sample_rate).astype(int)
        indices_mod = np.clip(indices_mod, 0, len(data) - 1)
        chorus = data[indices_mod]
        
        # Add reverb
        reverb_time = 1.5  # seconds
        reverb_samples = int(reverb_time * sample_rate)
        decay = np.exp(-6.0 * np.arange(reverb_samples) / reverb_samples)
        
        impulse = np.zeros(reverb_samples)
        impulse[0] = 1.0
        impulse *= decay
        
        data_reverb = signal.fftconvolve(data, impulse, mode='full')[:len(data)]
        
        # Mix dry, chorus, and reverb
        data = 0.3 * data + 0.2 * chorus + 0.5 * data_reverb
        
        return data
    
    def _apply_melancholic_processing(self, data, sample_rate):
        """Apply processing for melancholic mood."""
        # Bandpass filter focused on lower mids
        sos = signal.butter(2, [200, 2500], 'bp', fs=sample_rate, output='sos')
        data = signal.sosfilt(sos, data)
        
        # Long reverb with dark character
        reverb_time = 4.0  # seconds
        reverb_samples = int(reverb_time * sample_rate)
        
        # Use non-linear decay for darker reverb
        decay = np.exp(-4.0 * np.sqrt(np.arange(reverb_samples) / reverb_samples))
        
        impulse = np.zeros(reverb_samples)
        impulse[0] = 1.0
        impulse *= decay
        
        # Apply low-pass filter to the impulse response for darker reverb
        sos_dark = signal.butter(2, 2000, 'lp', fs=sample_rate, output='sos')
        impulse = signal.sosfilt(sos_dark, impulse)
        
        # Apply reverb through convolution
        data_reverb = signal.fftconvolve(data, impulse, mode='full')[:len(data)]
        
        # Create long, quiet delay
        delay_time = 0.75  # seconds
        delay_samples = int(delay_time * sample_rate)
        delay_gain = 0.25
        
        data_delay = np.zeros_like(data)
        data_delay[delay_samples:] = data[:-delay_samples] * delay_gain
        
        # Mix with emphasis on reverb for spaciousness
        data = 0.2 * data + 0.6 * data_reverb + 0.2 * data_delay
        
        return data
    
    def _freq_to_note_name(self, freq):
        """Convert frequency to the closest note name."""
        # A4 = 440 Hz
        A4 = 440.0
        C0 = A4 * pow(2, -4.75)  # C0 is 4.75 octaves below A4
        
        # Calculate the note number
        h = round(12.0 * np.log2(freq / C0))
        
        # Calculate octave and note
        octave = h // 12
        n = h % 12
        
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note_name = notes[n] + str(octave)
        
        return note_name

def main():
    parser = argparse.ArgumentParser(description='Generate ambient textures for raga-based lo-fi.')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Tanpura generator command
    tanpura_parser = subparsers.add_parser('tanpura', help='Generate tanpura-like drone')
    tanpura_parser.add_argument('--tonic', '-t', type=float, default=261.63, 
                             help='Tonic frequency in Hz (default: 261.63 Hz = C4)')
    tanpura_parser.add_argument('--note', '-n', help='Tonic note name (e.g., C4, G3) - overrides tonic frequency')
    tanpura_parser.add_argument('--duration', '-d', type=int, default=60, help='Duration in seconds')
    tanpura_parser.add_argument('--mood', '-m', default='peaceful', 
                             choices=['peaceful', 'devotional', 'contemplative', 'romantic', 'melancholic'],
                             help='Texture mood')
    tanpura_parser.add_argument('--output-dir', '-o', default='generated_textures', help='Output directory')
    
    # Process audio command
    process_parser = subparsers.add_parser('process', help='Process existing audio file')
    process_parser.add_argument('input_file', help='Input audio file')
    process_parser.add_argument('--mood', '-m', default='peaceful', 
                              choices=['peaceful', 'devotional', 'contemplative', 'romantic', 'melancholic'],
                              help='Texture mood')
    process_parser.add_argument('--output', '-o', help='Output file (default: auto-generated)')
    process_parser.add_argument('--output-dir', '-d', default='generated_textures', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize texture generator
    output_dir = args.output_dir
    generator = RagaTextureGenerator(output_dir)
    
    if args.command == 'tanpura':
        # Convert note name to frequency if provided
        if args.note:
            note_to_freq = {
                'C3': 130.81, 'C#3': 138.59, 'D3': 146.83, 'D#3': 155.56, 'E3': 164.81, 'F3': 174.61,
                'F#3': 185.00, 'G3': 196.00, 'G#3': 207.65, 'A3': 220.00, 'A#3': 233.08, 'B3': 246.94,
                'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13, 'E4': 329.63, 'F4': 349.23,
                'F#4': 369.99, 'G4': 392.00, 'G#4': 415.30, 'A4': 440.00, 'A#4': 466.16, 'B4': 493.88
            }
            if args.note in note_to_freq:
                tonic_freq = note_to_freq[args.note]
            else:
                print(f"Unknown note: {args.note}. Using default frequency.")
                tonic_freq = args.tonic
        else:
            tonic_freq = args.tonic
        
        generator.generate_tanpura_texture(
            tonic_freq=tonic_freq,
            duration=args.duration,
            mood=args.mood
        )
    
    elif args.command == 'process':
        output_file = args.output
        if output_file is None:
            timestamp = int(time.time())
            filename = os.path.splitext(os.path.basename(args.input_file))[0]
            output_file = os.path.join(output_dir, f"{filename}_{args.mood}_{timestamp}.wav")
        
        generator.process_audio_file(
            input_file=args.input_file,
            mood=args.mood,
            output_file=output_file
        )
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()