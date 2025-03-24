#!/usr/bin/env python3
"""
Raga Lo-Fi Toolkit - Main Application
------------------------------------
This module orchestrates the complete workflow from raga selection to lo-fi music generation,
integrating harmony analysis, AudioCraft bridge, and audio processing components.
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Import core components
from enhanced_raga_generator import EnhancedRagaGenerator
from lofi_stack_generator import LofiStackGenerator
from harmony_analyzer import HarmonyAnalyzer
from audiocraft_bridge import get_music_gen, AudioCraftBridge
from audio_processor import AudioProcessor, process_audio_file

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('raga_lofi_main')

# Default paths
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_SAMPLE_LIBRARY = "sample_library"
DEFAULT_CONFIG_FILE = "config.json"

class RagaLofiApp:
    """
    Main application class that orchestrates the raga-based lo-fi music generation workflow.
    Integrates all components: EnhancedRagaGenerator, HarmonyAnalyzer, AudioCraftBridge, and AudioProcessor.
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 output_dir: str = DEFAULT_OUTPUT_DIR,
                 sample_library: str = DEFAULT_SAMPLE_LIBRARY,
                 verbose: bool = False):
        """
        Initialize the Raga Lo-Fi Application.
        
        Args:
            config_file: Path to configuration file (JSON)
            output_dir: Directory for output files
            sample_library: Path to sample library
            verbose: Enable verbose output
        """
        self.output_dir = output_dir
        self.sample_library = sample_library
        self.verbose = verbose
        
        # Set up logging level based on verbose flag
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize core components
        self.stack_generator = LofiStackGenerator(
            sample_library_path=sample_library,
            output_dir=output_dir
        )
        
        self.raga_generator = self.stack_generator.raga_generator
        self.harmony_analyzer = HarmonyAnalyzer()
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor()
        
        # Initialize AudioCraft bridge on-demand only
        self._audiocraft_bridge = None
        
        logger.info(f"Raga Lo-Fi App initialized (Output: {output_dir}, Sample Library: {sample_library})")
    
    def _load_config(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Load application configuration from file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        # Default configuration
        default_config = {
            "audiocraft": {
                "model_size": "small",
                "device": "cpu",
                "model_path": os.path.expanduser("~/audiocraft")
            },
            "audio_processing": {
                "sample_rate": 44100,
                "bit_depth": 16,
                "channels": 2
            },
            "generation": {
                "default_bpm": 75,
                "default_key": "C",
                "default_length": 32,
                "default_lofi_style": "classic",
                "effect_intensity": 0.5
            }
        }
        
        # Try to load from file if provided
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    
                # Merge configs (shallow merge for now)
                for section, settings in user_config.items():
                    if section in default_config:
                        default_config[section].update(settings)
                    else:
                        default_config[section] = settings
                
                logger.info(f"Configuration loaded from {config_file}")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(f"Could not load config file: {e}. Using defaults.")
        
        return default_config
    
    def get_audiocraft_bridge(self) -> AudioCraftBridge:
        """
        Get or initialize the AudioCraft bridge.
        
        Returns:
            Initialized AudioCraftBridge object
        """
        if self._audiocraft_bridge is None:
            # Get parameters from config
            ac_config = self.config['audiocraft']
            
            # Initialize bridge
            self._audiocraft_bridge = get_music_gen(
                model_size=ac_config['model_size'],
                device=ac_config['device'],
                model_path=ac_config['model_path']
            )
            
        return self._audiocraft_bridge
    
    def list_available_ragas(self) -> None:
        """List all available ragas with their details."""
        self.stack_generator.list_available_ragas(self.raga_generator)
    
    def generate_complete_track(self, 
                              raga_id: Optional[str] = None,
                              melody_file: Optional[str] = None,
                              bpm: Optional[int] = None,
                              key: Optional[str] = None,
                              components: Optional[Dict[str, bool]] = None,
                              use_audiocraft: bool = False,
                              ai_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a complete raga-based lo-fi track.
        
        Args:
            raga_id: ID of the raga to use
            melody_file: Optional input melody file (MIDI or audio)
            bpm: Beats per minute
            key: Musical key
            components: Dictionary specifying which components to generate
            use_audiocraft: Whether to use AudioCraft for generation
            ai_prompt: Custom prompt for AudioCraft generation
            
        Returns:
            Dictionary with generation results
        """
        # Set default components if not provided
        if components is None:
            components = {
                "melody": True,
                "chords": True,
                "bass": True,
                "drums": True,
                "effects": True
            }
        
        # Set defaults from config if not provided
        gen_config = self.config['generation']
        if bpm is None:
            bpm = gen_config['default_bpm']
        if key is None:
            key = gen_config['default_key']
            
        # Configure stack generator
        self.stack_generator.set_bpm(bpm)
        self.stack_generator.set_key(key)
        
        # Generate based on input type
        if melody_file:
            # Generate from input melody
            logger.info(f"Generating track from melody: {melody_file}")
            result = self.stack_generator.generate_from_melody(
                melody_file=melody_file,
                identify_raga=True,
                components=components
            )
        elif raga_id:
            # Generate from specified raga
            logger.info(f"Generating track from raga: {raga_id}")
            result = self.stack_generator.generate_from_raga(
                raga_id=raga_id,
                length=gen_config['default_length'],
                components=components
            )
        else:
            # Use random raga
            logger.info("Generating track from random raga")
            available_ragas = self.raga_generator.list_available_ragas()
            if not available_ragas:
                logger.error("No ragas available in database")
                return None
            
            import random
            random_raga = random.choice(available_ragas)
            raga_id = random_raga['id']
            result = self.stack_generator.generate_from_raga(
                raga_id=raga_id,
                length=gen_config['default_length'],
                components=components
            )
        
        # Check result
        if not result:
            logger.error("Failed to generate track")
            return None
        
        # Perform harmony analysis if needed
        if components.get("melody", True) and "melody_midi" in result.get("files", {}):
            melody_file = result["files"]["melody_midi"]
            harmony_data = self.harmony_analyzer.analyze_midi(melody_file)
            
            if harmony_data:
                # Add harmony data to result
                result["harmony_analysis"] = harmony_data
                
                # Save harmony data
                harmony_file = os.path.join(
                    os.path.dirname(melody_file),
                    "harmony_analysis.json"
                )
                self.harmony_analyzer.export_results(harmony_file)
                result["files"]["harmony_analysis"] = harmony_file
        
        # Use AudioCraft if requested
        if use_audiocraft:
            try:
                # Get bridge
                bridge = self.get_audiocraft_bridge()
                
                # Determine prompt
                if not ai_prompt:
                    # Extract raga info for better prompt
                    raga_name = result.get("raga_name", "unknown")
                    mood = "relaxing"  # Default mood
                    
                    # Get mood from raga if available
                    if "raga_id" in result and result["raga_id"] in self.raga_generator.ragas_data:
                        raga = self.raga_generator.ragas_data[result["raga_id"]]
                        mood = raga.get("mood", "relaxing")
                    
                    ai_prompt = f"lofi {mood} music based on {raga_name} raga, with relaxing beats and ambient atmosphere"
                
                # Use harmony data if available
                harmony_data = result.get("harmony_analysis", None)
                reference_audio = None
                
                # If melody exists, use it as reference
                if "melody_midi" in result.get("files", {}) and os.path.exists(result["files"]["melody_midi"]):
                    reference_audio = result["files"]["melody_midi"]
                
                logger.info(f"Generating audio with AudioCraft: {ai_prompt}")
                
                # Determine generation approach
                if harmony_data:
                    # Generate with harmony-informed parameters
                    output = bridge.generate_from_analysis(
                        harmony_analysis=harmony_data,
                        duration=30,  # 30 seconds
                        variations=1,
                        enhancement_level=0.7,
                        raga_id=result.get("raga_id"),
                        reference_audio=reference_audio
                    )
                else:
                    # Generate with prompt only
                    output = bridge.generate(
                        prompts=ai_prompt,
                        duration=30  # 30 seconds
                    )
                
                # Save generated audio
                if output:
                    output_dir = os.path.dirname(result["files"].get("melody_midi", ""))
                    if not output_dir:
                        output_dir = self.output_dir
                    
                    ai_output_path = os.path.join(output_dir, "audiocraft_output.wav")
                    bridge.save_audio(output[0], ai_output_path)
                    
                    # Add to result
                    result["files"]["ai_generated"] = ai_output_path
                    logger.info(f"AudioCraft output saved to {ai_output_path}")
                    
                    # Process the AI output with lo-fi effects
                    processed_path = self._apply_lofi_effects(ai_output_path, harmony_data)
                    if processed_path:
                        result["files"]["ai_processed"] = processed_path
            
            except Exception as e:
                logger.error(f"Error using AudioCraft: {e}")
                # Continue without AudioCraft
        
        logger.info("Track generation completed successfully")
        return result
    
    def _apply_lofi_effects(self, audio_file: str, harmony_data: Optional[Dict] = None) -> Optional[str]:
        """
        Apply lo-fi effects to an audio file.
        
        Args:
            audio_file: Path to audio file
            harmony_data: Optional harmony analysis data
            
        Returns:
            Path to processed audio file
        """
        # Get settings from config
        lofi_style = self.config['generation']['default_lofi_style']
        effect_intensity = self.config['generation']['effect_intensity']
        
        # Generate output path
        output_file = os.path.join(
            os.path.dirname(audio_file),
            f"{os.path.splitext(os.path.basename(audio_file))[0]}_lofi.wav"
        )
        
        logger.info(f"Applying {lofi_style} lo-fi effects to {audio_file}")
        
        # Process the audio
        try:
            processed_path = process_audio_file(
                file_path=audio_file,
                harmony_data=harmony_data,
                lofi_style=lofi_style,
                effect_intensity=effect_intensity,
                output_path=output_file
            )
            
            if processed_path:
                logger.info(f"Lo-fi processed audio saved to {processed_path}")
                return processed_path
            
        except Exception as e:
            logger.error(f"Error applying lo-fi effects: {e}")
        
        return None
    
    def process_audio(self, 
                    audio_file: str,
                    harmony_file: Optional[str] = None,
                    lofi_style: Optional[str] = None,
                    effect_intensity: Optional[float] = None,
                    output_file: Optional[str] = None) -> Optional[str]:
        """
        Process an existing audio file with lo-fi effects.
        
        Args:
            audio_file: Path to audio file to process
            harmony_file: Optional path to harmony analysis JSON
            lofi_style: Lo-fi style to apply
            effect_intensity: Effect intensity (0.0-1.0)
            output_file: Output file path
            
        Returns:
            Path to processed audio file
        """
        # Use defaults from config if not provided
        if lofi_style is None:
            lofi_style = self.config['generation']['default_lofi_style']
        if effect_intensity is None:
            effect_intensity = self.config['generation']['effect_intensity']
        
        # Generate output path if not provided
        if output_file is None:
            output_file = os.path.join(
                self.output_dir,
                f"{os.path.splitext(os.path.basename(audio_file))[0]}_lofi.wav"
            )
        
        try:
            # First analyze the audio if harmony file not provided
            if not harmony_file:
                logger.info(f"Analyzing harmonic content of {audio_file}")
                harmony_data = self.harmony_analyzer.analyze_audio(audio_file)
                
                if harmony_data:
                    # Save harmony analysis
                    harmony_file = os.path.join(
                        os.path.dirname(output_file),
                        f"{os.path.splitext(os.path.basename(audio_file))[0]}_harmony.json"
                    )
                    self.harmony_analyzer.export_results(harmony_file)
            
            # Process the audio
            logger.info(f"Applying {lofi_style} lo-fi effects to {audio_file}")
            
            processed_path = process_audio_file(
                file_path=audio_file,
                harmony_data=harmony_file,
                lofi_style=lofi_style,
                effect_intensity=effect_intensity,
                output_path=output_file
            )
            
            if processed_path:
                logger.info(f"Lo-fi processed audio saved to {processed_path}")
                return processed_path
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
        
        return None
    
    def use_audiocraft(self, 
                     prompt: str,
                     reference_audio: Optional[str] = None,
                     duration: float = 30.0,
                     apply_lofi: bool = True,
                     output_file: Optional[str] = None) -> Optional[Dict[str, str]]:
        """
        Use AudioCraft to generate audio content.
        
        Args:
            prompt: Text prompt for AudioCraft
            reference_audio: Optional reference audio file
            duration: Duration in seconds
            apply_lofi: Whether to apply lo-fi effects to the result
            output_file: Output file path
            
        Returns:
            Dictionary with paths to generated files
        """
        try:
            # Get bridge
            bridge = self.get_audiocraft_bridge()
            
            # Generate output path if not provided
            if output_file is None:
                timestamp = int(time.time())
                output_file = os.path.join(
                    self.output_dir,
                    f"audiocraft_{timestamp}.wav"
                )
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Generate audio
            logger.info(f"Generating audio with AudioCraft: {prompt}")
            
            if reference_audio:
                if not os.path.exists(reference_audio):
                    logger.error(f"Reference audio file not found: {reference_audio}")
                    return None
                
                logger.info(f"Using reference audio: {reference_audio}")
                output = bridge.generate_with_reference(
                    prompts=prompt,
                    reference_audio=reference_audio,
                    duration=duration
                )
            else:
                output = bridge.generate(
                    prompts=prompt,
                    duration=duration
                )
            
            # Save generated audio
            if output:
                bridge.save_audio(output[0], output_file)
                logger.info(f"AudioCraft output saved to {output_file}")
                
                # Create result dictionary
                result = {
                    "original": output_file
                }
                
                # Apply lo-fi effects if requested
                if apply_lofi:
                    processed_path = self._apply_lofi_effects(output_file)
                    if processed_path:
                        result["processed"] = processed_path
                
                return result
            
        except Exception as e:
            logger.error(f"Error using AudioCraft: {e}")
        
        return None

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Raga Lo-Fi Toolkit - Generate raga-based lo-fi music'
    )
    
    # Main commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List ragas command
    list_parser = subparsers.add_parser('list-ragas', help='List all available ragas')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate a lo-fi track')
    
    # Generator source (mutually exclusive)
    source_group = generate_parser.add_mutually_exclusive_group()
    source_group.add_argument('--raga', help='Raga ID to use for generation')
    source_group.add_argument('--melody', help='Path to input melody file (MIDI or audio)')
    
    # Generation settings
    generate_parser.add_argument('--bpm', type=int, help='Beats per minute')
    generate_parser.add_argument('--key', help='Musical key (e.g., C, F#)')
    
    # Component toggles
    generate_parser.add_argument('--no-melody', action='store_true', help='Skip melody generation')
    generate_parser.add_argument('--no-chords', action='store_true', help='Skip chord generation')
    generate_parser.add_argument('--no-bass', action='store_true', help='Skip bass generation')
    generate_parser.add_argument('--no-drums', action='store_true', help='Skip drum selection')
    generate_parser.add_argument('--no-effects', action='store_true', help='Skip effect selection')
    
    # AudioCraft integration
    generate_parser.add_argument('--use-ai', action='store_true', help='Use AudioCraft for generation')
    generate_parser.add_argument('--ai-prompt', help='Custom prompt for AudioCraft generation')
    
    # Process audio command
    process_parser = subparsers.add_parser('process', help='Process existing audio with lo-fi effects')
    process_parser.add_argument('file', help='Audio file to process')
    process_parser.add_argument('--harmony', help='Path to harmony analysis JSON file')
    process_parser.add_argument('--style', choices=['classic', 'tape', 'vinyl', 'ambient'], 
                              help='Lo-fi style to apply')
    process_parser.add_argument('--intensity', type=float, help='Effect intensity (0.0-1.0)')
    process_parser.add_argument('-o', '--output', help='Output file path')
    
    # AudioCraft command
    audiocraft_parser = subparsers.add_parser('audiocraft', help='Generate audio with AudioCraft')
    audiocraft_parser.add_argument('prompt', help='Text prompt for generation')
    audiocraft_parser.add_argument('--reference', help='Reference audio file for conditioning')
    audiocraft_parser.add_argument('--duration', type=float, default=30.0, help='Duration in seconds')
    audiocraft_parser.add_argument('--no-lofi', action='store_true', help='Skip lo-fi processing')
    audiocraft_parser.add_argument('-o', '--output', help='Output file path')
    
    # Global options
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--output-dir', help='Output directory')
    parser.add_argument('--sample-library', help='Sample library path')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed output')
    
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    try:
        # Initialize application
        app = RagaLofiApp(
            config_file=args.config,
            output_dir=args.output_dir or DEFAULT_OUTPUT_DIR,
            sample_library=args.sample_library or DEFAULT_SAMPLE_LIBRARY,
            verbose=args.verbose
        )
        
        # Execute the requested command
        if args.command == 'list-ragas':
            app.list_available_ragas()
            
        elif args.command == 'generate':
            # Prepare components dict
            components = {
                "melody": not args.no_melody,
                "chords": not args.no_chords,
                "bass": not args.no_bass,
                "drums": not args.no_drums,
                "effects": not args.no_effects
            }
            
            # Generate track
            result = app.generate_complete_track(
                raga_id=args.raga,
                melody_file=args.melody,
                bpm=args.bpm,
                key=args.key,
                components=components,
                use_audiocraft=args.use_ai,
                ai_prompt=args.ai_prompt
            )
            
            if result:
                # Report success
                print("\n✅ Track generation successful!")
                print(f"Output directory: {os.path.dirname(result['files'].get('melody_midi', ''))}")
                
                if 'raga_id' in result and 'raga_name' in result:
                    print(f"Raga: {result['raga_name']} ({result['raga_id']})")
                
                # Print paths to generated files
                if args.verbose:
                    print("\nGenerated files:")
                    for category, files in result['files'].items():
                        if isinstance(files, dict):
                            print(f"\n{category.capitalize()}:")
                            for name, path in files.items():
                                print(f"- {name}: {os.path.basename(path)}")
                        else:
                            print(f"- {category}: {os.path.basename(files)}")
            else:
                print("\n❌ Track generation failed")
                
        elif args.command == 'process':
            # Process audio file
            result = app.process_audio(
                audio_file=args.file,
                harmony_file=args.harmony,
                lofi_style=args.style,
                effect_intensity=args.intensity,
                output_file=args.output
            )
            
            if result:
                print(f"\n✅ Audio processing successful!")
                print(f"Output file: {result}")
            else:
                print("\n❌ Audio processing failed")
                
        elif args.command == 'audiocraft':
            # Use AudioCraft
            result = app.use_audiocraft(
                prompt=args.prompt,
                reference_audio=args.reference,
                duration=args.duration,
                apply_lofi=not args.no_lofi,
                output_file=args.output
            )
            
            if result:
                print(f"\n✅ AudioCraft generation successful!")
                print(f"Generated file: {result['original']}")
                
                if 'processed' in result:
                    print(f"Processed file: {result['processed']}")
            else:
                print("\n❌ AudioCraft generation failed")
                
        else:
            # No command or unknown command
            print("Please specify a command. Run with --help for usage information.")
            
    except Exception as e:
        logger.error(f"Error in main application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()