#!/usr/bin/env python3
"""
Raga Lo-Fi Beat Generator CLI
----------------------------
Command-line interface for generating lo-fi beats based on Indian ragas.
"""

import os
import sys
import argparse
import json
import time
import random
from pprint import pprint

# Import the generator
from lofi_stack_generator import LofiStackGenerator

def list_available_ragas(generator):
    """List all available ragas with their details."""
    ragas = generator.raga_generator.list_available_ragas()
    
    if not ragas:
        print("No ragas available in the database.")
        return
    
    print("\nAvailable Ragas:")
    print("================")
    
    # Group by mood
    ragas_by_mood = {}
    for raga in ragas:
        mood = raga['mood']
        if mood not in ragas_by_mood:
            ragas_by_mood[mood] = []
        ragas_by_mood[mood].append(raga)
    
    # Print grouped by mood
    for mood, mood_ragas in sorted(ragas_by_mood.items()):
        print(f"\n{mood.capitalize()}:")
        print("-" * len(mood))
        
        for raga in sorted(mood_ragas, key=lambda x: x['name']):
            system = ""
            if 'system' in raga:
                system = f" ({raga['system'].capitalize()})"
            
            time_of_day = f", {raga['time']}" if 'time' in raga else ""
            print(f"- {raga['name']}{system}{time_of_day} [ID: {raga['id']}]")
    
    print("\nTo use a raga, run: python lofi_generator_cli.py generate --raga <raga_id>")

def generate_beat(args):
    """Generate a lo-fi beat with the given parameters."""
    # Initialize the generator
    generator = LofiStackGenerator(
        sample_library_path=args.sample_library,
        output_dir=args.output_dir
    )
    
    # Set BPM
    if args.bpm:
        generator.set_bpm(args.bpm)
    
    # Set key
    if args.key:
        generator.set_key(args.key)
    
    # Determine generation mode
    if args.raga:
        # Generate from raga
        if not generator.set_raga(args.raga):
            print(f"Error: Raga '{args.raga}' not found.")
            print("Use the 'list-ragas' command to see available ragas.")
            return
        
        print(f"\nGenerating lo-fi beat from raga: {args.raga}")
        result = generator.generate_from_raga(
            length=args.length,
            components={
                "melody": not args.no_melody,
                "chords": not args.no_chords,
                "bass": not args.no_bass,
                "drums": not args.no_drums,
                "effects": not args.no_effects
            }
        )
    
    elif args.melody:
        # Generate from input melody
        if not os.path.exists(args.melody):
            print(f"Error: Melody file not found: {args.melody}")
            return
        
        print(f"\nGenerating lo-fi beat from melody: {args.melody}")
        result = generator.generate_from_melody(
            args.melody,
            identify_raga=not args.no_analyze,
            components={
                "melody": False,  # Already provided
                "chords": not args.no_chords,
                "bass": not args.no_bass,
                "drums": not args.no_drums,
                "effects": not args.no_effects
            }
        )
    
    elif args.loop:
        # Generate from input loop
        if not os.path.exists(args.loop):
            print(f"Error: Loop file not found: {args.loop}")
            return
        
        print(f"\nGenerating lo-fi beat from loop: {args.loop}")
        result = generator.generate_from_loop(
            args.loop,
            components={
                "melody": not args.no_melody,
                "chords": not args.no_chords,
                "bass": not args.no_bass,
                "drums": not args.no_drums,
                "effects": not args.no_effects
            }
        )
    
    else:
        # Random raga generation
        available_ragas = generator.raga_generator.list_available_ragas()
        if not available_ragas:
            print("Error: No ragas available in the database.")
            return
        
        random_raga = random.choice(available_ragas)
        raga_id = random_raga['id']
        generator.set_raga(raga_id)
        
        print(f"\nGenerating lo-fi beat from random raga: {random_raga['name']} ({raga_id})")
        result = generator.generate_from_raga(
            length=args.length,
            components={
                "melody": not args.no_melody,
                "chords": not args.no_chords,
                "bass": not args.no_bass,
                "drums": not args.no_drums,
                "effects": not args.no_effects
            }
        )
    
    # Print result summary
    if result:
        print("\n✅ Beat generation successful!")
        print(f"Output directory: {os.path.dirname(result['files'].get('project', {}).get('instructions', ''))}")
        
        if 'raga_id' in result:
            print(f"Raga: {result['raga_name']} ({result['raga_id']})")
        
        if args.verbose:
            print("\nGeneration details:")
            # Convert file paths to relative paths for cleaner output
            simplified_result = {
                'raga_id': result.get('raga_id', 'N/A'),
                'raga_name': result.get('raga_name', 'N/A'),
                'bpm': result.get('bpm', 'N/A'),
                'key': result.get('key', 'N/A'),
                'files': {}
            }
            
            for category, files in result['files'].items():
                if isinstance(files, dict):
                    simplified_result['files'][category] = {
                        k: os.path.basename(v) for k, v in files.items()
                    }
                else:
                    simplified_result['files'][category] = os.path.basename(files)
            
            # Print the simplified result
            pprint(simplified_result)
    else:
        print("\n❌ Beat generation failed.")

def main():
    parser = argparse.ArgumentParser(description='Generate lo-fi beats based on Indian ragas.')
    
    # Main commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List ragas command
    list_parser = subparsers.add_parser('list-ragas', help='List all available ragas')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate a lo-fi beat')
    
    # Generator source (mutually exclusive)
    source_group = generate_parser.add_mutually_exclusive_group()
    source_group.add_argument('--raga', help='Raga ID to use for generation')
    source_group.add_argument('--melody', help='Path to input melody file (MIDI or audio)')
    source_group.add_argument('--loop', help='Path to input loop file (audio)')
    
    # Generation settings
    generate_parser.add_argument('--bpm', type=int, help='Beats per minute (default: 75)')
    generate_parser.add_argument('--key', help='Musical key (default: C)')
    generate_parser.add_argument('--length', type=int, default=32, help='Length of the melody in notes')
    
    # Component toggles
    generate_parser.add_argument('--no-melody', action='store_true', help='Skip melody generation')
    generate_parser.add_argument('--no-chords', action='store_true', help='Skip chord generation')
    generate_parser.add_argument('--no-bass', action='store_true', help='Skip bass generation')
    generate_parser.add_argument('--no-drums', action='store_true', help='Skip drum selection')
    generate_parser.add_argument('--no-effects', action='store_true', help='Skip effect selection')
    generate_parser.add_argument('--no-analyze', action='store_true', help='Skip raga analysis of input melody')
    
    # Output options
    generate_parser.add_argument('--output-dir', default='outputs', help='Output directory')
    generate_parser.add_argument('--sample-library', default='sample_library', help='Sample library path')
    generate_parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed output')
    
    args = parser.parse_args()
    
    # Execute the requested command
    if args.command == 'list-ragas':
        generator = LofiStackGenerator()
        list_available_ragas(generator)
    
    elif args.command == 'generate':
        generate_beat(args)
    
    else:
        # Default to showing help
        parser.print_help()

if __name__ == "__main__":
    main()