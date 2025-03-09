#!/usr/bin/env python3
"""
Raga Identification to Lo-Fi Workflow
------------------------------------
This script demonstrates how to integrate raga identification
with the lo-fi music generation workflow.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Import the necessary modules
from raga_identifier import RagaIdentifier
from melodic_pattern_generator import MelodicPatternGenerator


def identify_and_generate(file_path, output_dir="outputs", preprocess=True, 
                        creativity=0.5, gamaka_intensity=0.5, length=64):
    """
    Identify the raga in an audio file and generate lo-fi music based on it.
    
    Parameters:
    - file_path: Path to the audio file
    - output_dir: Directory for output files
    - preprocess: Whether to preprocess the audio for analysis
    - creativity: Creativity level for melody generation (0.0-1.0)
    - gamaka_intensity: Intensity of gamakas in generation (0.0-1.0)
    - length: Length of melody to generate
    
    Returns:
    - Dictionary with paths to generated files
    """
    print(f"\n==== ANALYZING RAGA IN: {file_path} ====\n")
    
    # Step 1: Identify the raga
    identifier = RagaIdentifier()
    analysis_results = identifier.identify_raga(file_path, preprocess=preprocess)
    
    if not analysis_results:
        print("Failed to identify raga. Cannot generate melody.")
        return None
    
    # Save the analysis results
    results_path = identifier.save_results()
    
    # Check if it's a mixed raga
    is_mixed = False
    if ('mixed_raga_analysis' in analysis_results and 
        analysis_results['mixed_raga_analysis'].get('is_mixed_raga', False)):
        is_mixed = True
        print("\nDetected a mixed raga composition. Will generate melodies for each identified raga.")
    
    # Create generator
    generator = MelodicPatternGenerator()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Store generation results
    generation_results = {
        'analysis_file': results_path,
        'source_file': file_path,
        'identified_ragas': [],
        'generated_files': []
    }
    
    # Generate for the primary raga
    if 'overall_results' in analysis_results and 'top_matches' in analysis_results['overall_results']:
        primary_raga = analysis_results['overall_results']['top_matches'][0]
        raga_id = primary_raga['raga_id']
        raga_name = primary_raga['raga_name']
        
        print(f"\n==== GENERATING LO-FI MUSIC FOR PRIMARY RAGA: {raga_name} ====\n")
        
        # Set the raga
        if generator.set_current_raga(raga_id):
            # Generate melody
            result = generator.generate_complete_track(
                length=length,
                creativity=creativity,
                gamaka_intensity=gamaka_intensity
            )
            
            if result:
                generation_results['identified_ragas'].append({
                    'raga_id': raga_id,
                    'raga_name': raga_name,
                    'confidence': primary_raga['confidence'],
                    'is_primary': True,
                    'files': result
                })
                generation_results['generated_files'].extend([
                    result['melody'],
                    result['accompaniment']
                ])
                
                print(f"Generated files for {raga_name}:")
                print(f"  Melody: {result['melody']}")
                print(f"  Accompaniment: {result['accompaniment']}")
    
    # Generate for mixed ragas if detected
    if is_mixed and 'segment_results' in analysis_results:
        # Get unique ragas from segments
        segment_ragas = {}
        
        for segment in analysis_results['segment_results']:
            if 'top_matches' in segment and segment['top_matches']:
                top_match = segment['top_matches'][0]
                raga_id = top_match['raga_id']
                
                # Skip if same as primary raga
                if raga_id == primary_raga['raga_id']:
                    continue
                
                # Skip if already processed
                if raga_id in segment_ragas:
                    continue
                
                segment_ragas[raga_id] = top_match
        
        # Generate for each unique secondary raga
        for raga_id, raga_info in segment_ragas.items():
            raga_name = raga_info['raga_name']
            
            print(f"\n==== GENERATING LO-FI MUSIC FOR SECONDARY RAGA: {raga_name} ====\n")
            
            # Set the raga
            if generator.set_current_raga(raga_id):
                # Generate melody - use higher creativity for secondary ragas
                result = generator.generate_complete_track(
                    length=length // 2,  # Shorter for secondary ragas
                    creativity=min(creativity + 0.2, 0.9),
                    gamaka_intensity=gamaka_intensity
                )
                
                if result:
                    generation_results['identified_ragas'].append({
                        'raga_id': raga_id,
                        'raga_name': raga_name,
                        'confidence': raga_info['confidence'],
                        'is_primary': False,
                        'files': result
                    })
                    generation_results['generated_files'].extend([
                        result['melody'],
                        result['accompaniment']
                    ])
                    
                    print(f"Generated files for {raga_name}:")
                    print(f"  Melody: {result['melody']}")
                    print(f"  Accompaniment: {result['accompaniment']}")
    
    # Save generation results
    summary_path = os.path.join(output_dir, f"generation_summary_{Path(file_path).stem}.json")
    with open(summary_path, 'w') as f:
        json.dump(generation_results, f, indent=2)
    
    print(f"\nGeneration summary saved to: {summary_path}")
    
    return generation_results


def main():
    """Main function to run the workflow from command line."""
    parser = argparse.ArgumentParser(
        description='Identify ragas and generate lo-fi music')
    
    # Input options
    parser.add_argument('file', help='Path to audio file for analysis')
    
    # Output options
    parser.add_argument('-o', '--output-dir', default='outputs', 
                      help='Directory for output files')
    
    # Preprocessing options
    parser.add_argument('--no-preprocess', action='store_false', dest='preprocess',
                      help='Disable audio preprocessing')
    
    # Generation options
    parser.add_argument('-c', '--creativity', type=float, default=0.5,
                      help='Creativity level for melody generation (0.0-1.0)')
    parser.add_argument('-g', '--gamaka', type=float, default=0.5,
                      help='Intensity of gamakas in generation (0.0-1.0)')
    parser.add_argument('-l', '--length', type=int, default=64,
                      help='Length of melody to generate')
    
    args = parser.parse_args()
    
    # Run the workflow
    identify_and_generate(
        args.file,
        output_dir=args.output_dir,
        preprocess=args.preprocess,
        creativity=args.creativity,
        gamaka_intensity=args.gamaka,
        length=args.length
    )


if __name__ == "__main__":
    main()