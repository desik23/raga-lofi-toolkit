#!/usr/bin/env python3
"""
AudioCraft Bridge for Raga-Lofi Integration
------------------------------------------
Provides a safe bridge to AudioCraft's MusicGen for generating audio content
compatible with raga-based harmony analysis and melody generation.
"""

import os
import sys
import time
import json
import tempfile
import logging
import pathlib
import warnings
import importlib.util
from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('audiocraft_bridge')

# Constants for default model paths
DEFAULT_MODEL_PATH = os.path.expanduser("~/audiocraft")
MODEL_SIZES = ["small", "medium", "large", "large_melody"]
DEFAULT_SIZE = "small"
DEFAULT_DEVICE = "cpu"  # Can be "cuda", "mps" (for Apple Silicon), or "cpu"

# Mapping of raga moods to prompt descriptors
RAGA_MOOD_DESCRIPTORS = {
    "peaceful": [
        "peaceful ambient lofi beats", 
        "calm meditation music", 
        "relaxing instrumental background"
    ],
    "devotional": [
        "spiritual ambient music", 
        "reverent lofi melody", 
        "sacred meditation soundscape"
    ],
    "melancholic": [
        "melancholic downtempo beats", 
        "sad lofi piano", 
        "nostalgic ambient music"
    ],
    "romantic": [
        "romantic piano melody", 
        "intimate acoustic instrumental", 
        "warm lofi love theme"
    ],
    "contemplative": [
        "thoughtful ambient soundscape", 
        "introspective minimal music", 
        "philosophical lofi beats"
    ],
    "joyful": [
        "uplifting melodic lofi", 
        "cheerful ambient instrumental", 
        "playful beats with piano"
    ],
    "energetic": [
        "rhythmic fusion beats", 
        "dynamic world music blend", 
        "energetic instrumental with percussion"
    ]
}

class ModelNotFoundError(Exception):
    """Exception raised when AudioCraft model cannot be found or loaded."""
    pass

class AudioCraftBridge:
    """
    Bridge to safely load and interact with AudioCraft's MusicGen models.
    Handles dependencies gracefully and provides integration with raga analysis.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 model_size: str = DEFAULT_SIZE,
                 device: str = DEFAULT_DEVICE,
                 cache_dir: Optional[str] = None):
        """
        Initialize the AudioCraft bridge.
        
        Args:
            model_path: Path to the AudioCraft installation (default: ~/audiocraft)
            model_size: Model size to use (small, medium, large)
            device: Device to run inference on (cuda, mps, cpu)
            cache_dir: Directory to cache model weights
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.device = device
        self.model_size = model_size
        self.cache_dir = cache_dir
        
        # Check if TorchAudio and AudioCraft are available
        self.torch_available = self._is_torch_available()
        self.torchaudio_available = self._is_torchaudio_available()
        self.audiocraft_available = self._is_audiocraft_available()
        
        # Initialize references to lazy-loaded components
        self._music_gen = None
        self._torch = None
        self._torchaudio = None
        self._models_loaded = False
        
        # Attempt to load dependencies if available
        if self.torch_available and self.torchaudio_available:
            try:
                import torch
                self._torch = torch
                import torchaudio
                self._torchaudio = torchaudio
            except ImportError:
                logger.warning("Failed to import torch or torchaudio even though they're available.")
    
    def _is_torch_available(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def _is_torchaudio_available(self) -> bool:
        """Check if TorchAudio is available."""
        try:
            import torchaudio
            return True
        except ImportError:
            return False
    
    def _is_audiocraft_available(self) -> bool:
        """Check if AudioCraft is available or can be added to path."""
        # First check if it's already installed
        if importlib.util.find_spec("audiocraft") is not None:
            return True
        
        # Check if it's available at the specified path
        if os.path.exists(self.model_path):
            audiocraft_init = os.path.join(self.model_path, "audiocraft", "__init__.py")
            if os.path.exists(audiocraft_init):
                return True
        
        return False
    
    def _add_audiocraft_to_path(self) -> bool:
        """Add AudioCraft to the Python path if available locally."""
        if not os.path.exists(self.model_path):
            logger.warning(f"AudioCraft path does not exist: {self.model_path}")
            return False
        
        if self.model_path not in sys.path:
            logger.info(f"Adding AudioCraft to Python path: {self.model_path}")
            sys.path.insert(0, self.model_path)
            
            # Verify if it's now importable
            if importlib.util.find_spec("audiocraft") is not None:
                return True
            else:
                logger.warning(f"Added {self.model_path} to Python path, but audiocraft is still not importable")
                sys.path.remove(self.model_path)
                return False
        
        return True
    
    def _load_models(self) -> bool:
        """
        Load AudioCraft models if available.
        Returns True if models are loaded successfully.
        """
        if self._models_loaded:
            return True
        
        # Check if dependencies are available
        if not self.torch_available:
            logger.error("PyTorch not found. Please install PyTorch and TorchAudio.")
            return False
        
        if not self.torchaudio_available:
            logger.error("TorchAudio not found. Please install TorchAudio.")
            return False
        
        # Try to add AudioCraft to path if needed
        if not self.audiocraft_available and not self._add_audiocraft_to_path():
            logger.error(f"AudioCraft not found at {self.model_path}. Please install it or set the correct path.")
            return False
        
        # Now try to import AudioCraft and load the models
        try:
            # Suppress warnings during model loading
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Import AudioCraft MusicGen
                from audiocraft.models import MusicGen
                
                # Load the model
                if self.model_size not in MODEL_SIZES:
                    logger.warning(f"Model size {self.model_size} not recognized, using {DEFAULT_SIZE}")
                    self.model_size = DEFAULT_SIZE
                
                logger.info(f"Loading MusicGen model: {self.model_size} on {self.device}...")
                
                self._music_gen = MusicGen.get_pretrained(
                    self.model_size, 
                    device=self.device,
                    cache_dir=self.cache_dir
                )
                
                self._models_loaded = True
                logger.info("MusicGen model loaded successfully.")
                return True
                
        except Exception as e:
            logger.error(f"Failed to load AudioCraft: {str(e)}")
            return False
    
    def generate(self, 
                prompts: Union[str, List[str]], 
                duration: float = 10.0,
                progress: bool = True) -> List[np.ndarray]:
        """
        Generate audio from text prompts.
        
        Args:
            prompts: Text prompt or list of prompts
            duration: Duration of generated audio in seconds
            progress: Whether to show progress bar during generation
            
        Returns:
            List of numpy arrays containing generated audio samples
        """
        if not self._load_models():
            raise ModelNotFoundError("Failed to load MusicGen models")
        
        try:
            # Convert single prompt to list
            if isinstance(prompts, str):
                prompts = [prompts]
            
            # Set generation parameters
            logger.info(f"Generating audio for {len(prompts)} prompt(s), duration: {duration}s")
            self._music_gen.set_generation_params(
                duration=duration,
                use_sampling=True,
                top_k=250,
                top_p=0.0,
                temperature=1.0,
                cfg_coef=3.0
            )
            
            # Generate audio
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Show progress if requested
                if progress:
                    import tqdm
                    print(f"Generating audio with MusicGen ({self.model_size})...")
                    for i, prompt in enumerate(prompts):
                        print(f"Prompt {i+1}: {prompt}")
                    
                    # Create progress bar based on duration
                    with tqdm.tqdm(total=100, desc="Generating") as pbar:
                        start_time = time.time()
                        output = self._music_gen.generate(prompts)
                        
                        # Update progress bar
                        generation_time = time.time() - start_time
                        pbar.update(100)
                        
                    print(f"Generation completed in {generation_time:.1f} seconds.")
                else:
                    output = self._music_gen.generate(prompts)
            
            # Convert outputs to numpy arrays
            return [wav.detach().cpu().numpy() for wav in output]
            
        except Exception as e:
            logger.error(f"Error during audio generation: {str(e)}")
            raise
    
    def generate_with_reference(self,
                               prompts: Union[str, List[str]],
                               reference_audio: Union[str, np.ndarray],
                               duration: float = 10.0,
                               progress: bool = True) -> List[np.ndarray]:
        """
        Generate audio conditioned on both text prompt and reference audio.
        
        Args:
            prompts: Text prompt or list of prompts
            reference_audio: Path to reference audio file or numpy array
            duration: Duration of generated audio in seconds
            progress: Whether to show progress bar during generation
            
        Returns:
            List of numpy arrays containing generated audio samples
        """
        if not self._load_models():
            raise ModelNotFoundError("Failed to load MusicGen models")
        
        # Check if model supports melody conditioning
        if "melody" not in self.model_size and self.model_size != "large_melody":
            logger.warning(
                f"Model {self.model_size} does not support melody conditioning. "
                "Use 'melody' or 'large_melody' model instead."
            )
            return self.generate(prompts, duration, progress)
        
        try:
            # Convert single prompt to list
            if isinstance(prompts, str):
                prompts = [prompts]
                
            # Process reference audio
            if isinstance(reference_audio, str):
                # Load audio file
                if not os.path.exists(reference_audio):
                    logger.error(f"Reference audio file not found: {reference_audio}")
                    return self.generate(prompts, duration, progress)
                
                # Load audio using torchaudio
                melody, sr = self._torchaudio.load(reference_audio)
                
                # Convert to mono if needed
                if melody.shape[0] > 1:
                    melody = torch.mean(melody, dim=0, keepdim=True)
                
                # Resample if necessary (MusicGen expects 44.1kHz)
                if sr != 44100:
                    melody = self._torchaudio.functional.resample(melody, sr, 44100)
                
            elif isinstance(reference_audio, np.ndarray):
                # Convert numpy array to torch tensor
                if len(reference_audio.shape) == 1:
                    # Mono audio, add channel dimension
                    melody = self._torch.from_numpy(reference_audio).unsqueeze(0)
                else:
                    # Already has channel dimension
                    melody = self._torch.from_numpy(reference_audio)
                    
                    # Convert to mono if needed
                    if melody.shape[0] > 1:
                        melody = torch.mean(melody, dim=0, keepdim=True)
            else:
                logger.error(f"Unsupported reference audio type: {type(reference_audio)}")
                return self.generate(prompts, duration, progress)
            
            # Set generation parameters
            logger.info(f"Generating audio with reference for {len(prompts)} prompt(s), duration: {duration}s")
            self._music_gen.set_generation_params(
                duration=duration,
                use_sampling=True,
                top_k=250,
                top_p=0.0,
                temperature=1.0,
                cfg_coef=3.0
            )
            
            # Generate audio
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Show progress if requested
                if progress:
                    import tqdm
                    print(f"Generating audio with MusicGen ({self.model_size}) and reference...")
                    for i, prompt in enumerate(prompts):
                        print(f"Prompt {i+1}: {prompt}")
                    
                    # Create progress bar based on duration
                    with tqdm.tqdm(total=100, desc="Generating") as pbar:
                        start_time = time.time()
                        
                        # Set melody conditioning
                        self._music_gen.set_melody_conditioning(melody)
                        output = self._music_gen.generate_with_chroma(prompts)
                        
                        # Update progress bar
                        generation_time = time.time() - start_time
                        pbar.update(100)
                        
                    print(f"Generation completed in {generation_time:.1f} seconds.")
                else:
                    # Set melody conditioning
                    self._music_gen.set_melody_conditioning(melody)
                    output = self._music_gen.generate_with_chroma(prompts)
            
            # Convert outputs to numpy arrays
            return [wav.detach().cpu().numpy() for wav in output]
            
        except Exception as e:
            logger.error(f"Error during reference-based audio generation: {str(e)}")
            raise
    
    def generate_from_analysis(self,
                              harmony_analysis: Dict[str, Any],
                              duration: float = 30.0,
                              variations: int = 1,
                              enhancement_level: float = 0.5,
                              raga_id: Optional[str] = None,
                              reference_audio: Optional[str] = None) -> List[np.ndarray]:
        """
        Generate audio based on harmony analysis results.
        
        Args:
            harmony_analysis: Dictionary containing harmony analysis results
            duration: Duration of generated audio in seconds
            variations: Number of variations to generate
            enhancement_level: Level of enhancement for prompt construction (0.0-1.0)
            raga_id: Optional raga ID for additional context
            reference_audio: Optional reference audio for conditioning
            
        Returns:
            List of numpy arrays containing generated audio samples
        """
        if not harmony_analysis:
            logger.error("No harmony analysis provided")
            return []
        
        # Extract relevant information from harmony analysis
        prompts = self._construct_prompts_from_analysis(
            harmony_analysis, 
            variations, 
            enhancement_level,
            raga_id
        )
        
        logger.info(f"Generated {len(prompts)} prompts from harmony analysis")
        
        # Generate audio with reference if provided, otherwise without
        if reference_audio:
            return self.generate_with_reference(
                prompts=prompts,
                reference_audio=reference_audio,
                duration=duration,
                progress=True
            )
        else:
            return self.generate(
                prompts=prompts,
                duration=duration,
                progress=True
            )
    
    def _construct_prompts_from_analysis(self,
                                       harmony_analysis: Dict[str, Any],
                                       variations: int = 1,
                                       enhancement_level: float = 0.5,
                                       raga_id: Optional[str] = None) -> List[str]:
        """
        Construct text prompts based on harmony analysis.
        
        Args:
            harmony_analysis: Dictionary containing harmony analysis results
            variations: Number of variations to generate
            enhancement_level: Level of enhancement for prompt construction (0.0-1.0)
            raga_id: Optional raga ID for additional context
            
        Returns:
            List of text prompts
        """
        prompts = []
        
        # Get chord progression from analysis
        chord_progression = []
        if 'chord_progression' in harmony_analysis:
            chord_progression = harmony_analysis['chord_progression']
        elif 'predominant_chords' in harmony_analysis:
            # Use predominant chords if progression not available
            chord_progression = [{"root_name": chord, "type": ""} 
                               for chord, _ in harmony_analysis['predominant_chords']]
        
        # Get raga info if available
        raga_name = None
        raga_mood = None
        raga_time = None
        
        # Check if we have raga info in the harmony analysis
        if 'raga_info' in harmony_analysis and raga_id:
            raga_info = harmony_analysis['raga_info']
            raga_name = raga_info.get('name')
            raga_mood = raga_info.get('mood')
            raga_time = raga_info.get('time')
        
        # Try to load raga info from model's raga data if we have raga ID
        if not raga_name and raga_id:
            if hasattr(self, 'harmony_analyzer'):
                if hasattr(self.harmony_analyzer, 'ragas') and raga_id in self.harmony_analyzer.ragas:
                    raga = self.harmony_analyzer.ragas[raga_id]
                    raga_name = raga.get('name')
                    raga_mood = raga.get('mood')
                    raga_time = raga.get('time')
        
        # Basic descriptors for lo-fi music
        base_descriptors = [
            "lofi hip hop beat",
            "chill instrumental beat",
            "relaxing background music",
            "ambient lofi instrumental",
            "downtempo beats with piano",
            "mellow lofi with soft drums"
        ]
        
        # Add mood-based descriptors if available
        mood_descriptors = []
        if raga_mood and raga_mood in RAGA_MOOD_DESCRIPTORS:
            mood_descriptors = RAGA_MOOD_DESCRIPTORS[raga_mood]
        
        # Add time-based modifiers
        time_modifiers = []
        if raga_time:
            if "morning" in raga_time.lower():
                time_modifiers = ["morning", "sunrise", "awakening"]
            elif "afternoon" in raga_time.lower():
                time_modifiers = ["afternoon", "midday", "bright"]
            elif "evening" in raga_time.lower():
                time_modifiers = ["evening", "sunset", "dusk"]
            elif "night" in raga_time.lower():
                time_modifiers = ["night", "moonlit", "nocturnal"]
        
        # Combine chord information if available
        chord_description = ""
        if chord_progression:
            # Format chord progression for prompt
            chord_names = []
            for chord in chord_progression[:4]:  # Limit to 4 chords to keep prompt reasonable
                if 'root_name' in chord and 'type' in chord:
                    chord_names.append(f"{chord['root_name']} {chord['type']}".strip())
                elif 'root_name' in chord:
                    chord_names.append(chord['root_name'])
            
            if chord_names:
                chord_description = f" with {', '.join(chord_names)} chord progression"
        
        # Generate variations
        for i in range(variations):
            # Base prompt components
            base = base_descriptors[i % len(base_descriptors)]
            
            # Add mood if available
            mood = ""
            if mood_descriptors:
                mood = mood_descriptors[i % len(mood_descriptors)]
            
            # Add time context if available
            time_context = ""
            if time_modifiers:
                time_context = time_modifiers[i % len(time_modifiers)]
            
            # Raga name for context
            raga_context = ""
            if raga_name:
                raga_context = f" inspired by {raga_name} raga"
            
            # Combine components based on enhancement level
            if enhancement_level < 0.3:
                # Minimal prompt
                prompt = f"{base}{chord_description}"
            elif enhancement_level < 0.7:
                # Medium enhancement
                if mood:
                    prompt = f"{mood}{chord_description}{raga_context}"
                else:
                    prompt = f"{base} {time_context}{chord_description}{raga_context}"
            else:
                # Full enhancement
                if mood and time_context:
                    prompt = f"{mood} {time_context}{chord_description}{raga_context}"
                elif mood:
                    prompt = f"{mood}{chord_description}{raga_context}"
                else:
                    prompt = f"{base} {time_context}{chord_description}{raga_context}"
            
            prompts.append(prompt.strip())
        
        return prompts
    
    def save_audio(self, 
                  audio: np.ndarray, 
                  output_path: str, 
                  sample_rate: int = 44100) -> str:
        """
        Save generated audio to a file.
        
        Args:
            audio: Numpy array containing audio samples
            output_path: Path to save the audio file
            sample_rate: Sample rate of the audio
            
        Returns:
            Path to the saved audio file
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Convert to torch tensor if needed
            if not self._torch:
                import torch
                self._torch = torch
                
            if not self._torchaudio:
                import torchaudio
                self._torchaudio = torchaudio
            
            # Convert numpy array to torch tensor if needed
            if isinstance(audio, np.ndarray):
                audio_tensor = self._torch.from_numpy(audio)
            else:
                audio_tensor = audio
            
            # Save audio
            self._torchaudio.save(
                output_path, 
                audio_tensor, 
                sample_rate
            )
            
            logger.info(f"Audio saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving audio: {str(e)}")
            return ""
    
    def available_model_sizes(self) -> List[str]:
        """Return available model sizes."""
        return MODEL_SIZES
    
    def is_model_available(self) -> bool:
        """Check if MusicGen model is available."""
        return self._load_models()
    
    def set_model_size(self, model_size: str) -> bool:
        """
        Change the model size.
        
        Args:
            model_size: New model size to use
        
        Returns:
            True if successful
        """
        if model_size not in MODEL_SIZES:
            logger.warning(f"Invalid model size {model_size}, using {DEFAULT_SIZE}")
            model_size = DEFAULT_SIZE
        
        # Only reload if different from current size
        if self.model_size != model_size:
            self.model_size = model_size
            self._models_loaded = False
            return self._load_models()
        
        return True


def get_music_gen(model_size: str = DEFAULT_SIZE, 
                 device: str = None,
                 model_path: str = None,
                 cache_dir: str = None) -> AudioCraftBridge:
    """
    Convenience function to get an AudioCraftBridge instance.
    
    Args:
        model_size: Model size to use (small, medium, large)
        device: Device to run inference on (cuda, mps, cpu)
        model_path: Path to AudioCraft installation
        cache_dir: Directory to cache model weights
        
    Returns:
        AudioCraftBridge instance
    """
    # Automatically select device if not specified
    if device is None:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"
    
    bridge = AudioCraftBridge(
        model_size=model_size,
        device=device,
        model_path=model_path,
        cache_dir=cache_dir
    )
    
    return bridge


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AudioCraft bridge for Raga-Lofi Integration')
    parser.add_argument('--prompt', type=str, help='Text prompt for audio generation')
    parser.add_argument('--reference', type=str, help='Reference audio file for conditioning')
    parser.add_argument('--output', type=str, default='generated_audio.wav', help='Output file')
    parser.add_argument('--model', type=str, default=DEFAULT_SIZE, choices=MODEL_SIZES, help='Model size')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda, mps, cpu)')
    parser.add_argument('--duration', type=float, default=10.0, help='Duration in seconds')
    parser.add_argument('--check-only', action='store_true', help='Only check if models are available')
    
    args = parser.parse_args()
    
    # Create the bridge
    bridge = get_music_gen(model_size=args.model, device=args.device)
    
    # Check if models are available
    if args.check_only:
        available = bridge.is_model_available()
        print(f"AudioCraft MusicGen ({args.model}) availability: {'Available' if available else 'Unavailable'}")
        sys.exit(0)
    
    # Check if we have a prompt
    if not args.prompt:
        print("No prompt provided. Use --prompt to specify text for generation.")
        sys.exit(1)
    
    # Generate audio
    try:
        if args.reference:
            # Generate with reference
            if not os.path.exists(args.reference):
                print(f"Reference file not found: {args.reference}")
                sys.exit(1)
                
            print(f"Generating audio with reference from prompt: '{args.prompt}'")
            outputs = bridge.generate_with_reference(
                prompts=args.prompt,
                reference_audio=args.reference,
                duration=args.duration
            )
        else:
            # Generate from prompt only
            print(f"Generating audio from prompt: '{args.prompt}'")
            outputs = bridge.generate(
                prompts=args.prompt,
                duration=args.duration
            )
        
        # Save the output
        if outputs:
            bridge.save_audio(outputs[0], args.output)
            print(f"Audio saved to {args.output}")
        else:
            print("No audio generated")
            
    except Exception as e:
        print(f"Error during audio generation: {str(e)}")
        sys.exit(1)