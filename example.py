#!/usr/bin/env python3
"""
Example usage of the Raga Lo-Fi Toolkit.
"""

import os
import sys
from src.enhanced_raga_generator import EnhancedRagaGenerator
from src.harmony_analyzer import HarmonyAnalyzer
from src.audio_processor import AudioProcessor

def example_raga_generation():
    """Example of basic raga melody generation."""
    print("Generating raga-based melody...")
    
    # Initialize the raga generator
    generator = EnhancedRagaGenerator()
    
    # List available ragas
    print("\nAvailable Ragas:")
    for raga in generator.list_available_ragas()[:10]:  # Show just the first 5
        print(f"- {raga['name']} ({raga['id']}): {raga['mood']}, {raga['time']}")
    
    # Choose a raga for demonstration
    raga_id = 'shankarabharanam'  
    
    try:
        # Generate a melody
        print(f"\nGenerating melody for {raga_id} raga...")
        melody_file = generator.generate_melody(raga_id, length=16, bpm=75)
        
        print(f"Melody generated: {melody_file}")
        
        # Analyze the melody
        print("\nAnalyzing harmonic content...")
        harmony_analyzer = HarmonyAnalyzer()
        harmony_data = harmony_analyzer.analyze_midi(melody_file)
        
        if harmony_data:
            print("\nHarmony Analysis:")
            if 'predominant_chords' in harmony_data:
                print("Predominant Chords:")
                for chord, count in harmony_data['predominant_chords']:
                    print(f"  {chord}: {count}")
            
            # Save the analysis
            harmony_file = os.path.join(os.path.dirname(melody_file), "harmony_analysis.json")
            harmony_analyzer.export_results(harmony_file)
            print(f"Harmony analysis saved to: {harmony_file}")
        
        print("\nExample completed successfully!")
        return melody_file, harmony_file
        
    except Exception as e:
        print(f"Error in example: {e}")
        return None, None

def example_audio_processing(audio_file=None, harmony_file=None):
    """Example of audio processing with lo-fi effects."""
    if audio_file is None:
        # Use a test file
        print("No audio file provided. Skipping audio processing example.")
        return
    
    print(f"\nProcessing audio file: {audio_file}")
    
    try:
        # Initialize audio processor
        processor = AudioProcessor()
        
        # Load audio
        if processor.load_audio(audio_file):
            print("Audio loaded successfully")
            
            # Load harmony data if available
            if harmony_file and os.path.exists(harmony_file):
                processor.load_harmony_data(harmony_file)
                print("Harmony data loaded")
            
            # Apply lo-fi effects
            lofi_style = 'classic'
            print(f"Applying {lofi_style} lo-fi effects...")
            processor.apply_lofi_effects(lofi_style, effect_intensity=0.6)
            
            # Save processed audio
            output_file = os.path.join(
                os.path.dirname(audio_file),
                f"{os.path.splitext(os.path.basename(audio_file))[0]}_processed.wav"
            )
            
            result = processor.save_audio(output_file)
            if result:
                print(f"Processed audio saved to: {result}")
            
            # Export effects chain
            chain_file = processor.export_effects_chain()
            if chain_file:
                print(f"Effects chain exported to: {chain_file}")
        
    except Exception as e:
        print(f"Error in audio processing example: {e}")

if __name__ == "__main__":
    print("Raga Lo-Fi Toolkit - Example Usage")
    print("==================================")
    
    # Run the raga generation example
    melody_file, harmony_file = example_raga_generation()
    
    # Run the audio processing example if we have a file
    if melody_file:
        print("\nNote: Audio processing would normally use an audio file, not MIDI.")
        print("To process actual audio, convert the MIDI to audio first.")
    
    print("\nFor full functionality, run the main application:")
    print("python src/main.py --help")