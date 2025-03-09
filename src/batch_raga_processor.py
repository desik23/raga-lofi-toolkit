#!/usr/bin/env python3
"""
Batch Raga Processor
-------------------
Batch processes audio files to identify ragas and/or generate lo-fi tracks.
"""

import os
import sys
import json
import csv
import time
import argparse
import concurrent.futures
from pathlib import Path
from tqdm import tqdm

# Import the necessary modules
from raga_identifier import RagaIdentifier
from melodic_pattern_generator import MelodicPatternGenerator
from raga_id_to_lofi import identify_and_generate


def process_file(file_path, output_dir="outputs", identify_only=False, preprocess=True, 
               creativity=0.5, gamaka_intensity=0.5, length=64, save_results=True):
    """
    Process a single file to identify raga and optionally generate lo-fi.
    
    Parameters:
    - file_path: Path to the audio file
    - output_dir: Directory for output files
    - identify_only: Whether to only identify the raga (no generation)
    - preprocess: Whether to preprocess the audio
    - creativity: Creativity level for melody generation (0.0-1.0)
    - gamaka_intensity: Intensity of gamakas in generation (0.0-1.0)
    - length: Length of melody to generate
    - save_results: Whether to save results to a file
    
    Returns:
    - Dictionary with processing results
    """
    results = {
        'file_path': file_path,
        'success': False,
        'identified_ragas': [],
        'generated_files': [],
        'error': None
    }
    
    try:
        # Create file-specific output directory
        file_name = Path(file_path).stem
        file_output_dir = os.path.join(output_dir, file_name)
        os.makedirs(file_output_dir, exist_ok=True)
        
        # Identify raga
        identifier = RagaIdentifier()
        analysis_results = identifier.identify_raga(file_path, preprocess=preprocess)
        
        if not analysis_results:
            results['error'] = "Failed to identify raga."
            return results
        
        # Save the analysis results
        results_path = identifier.save_results(
            os.path.join(file_output_dir, f"{file_name}_raga_analysis.json")
        )
        
        # Extract identified ragas
        if 'overall_results' in analysis_results and 'top_matches' in analysis_results['overall_results']:
            for match in analysis_results['overall_results']['top_matches']:
                results['identified_ragas'].append({
                    'raga_id': match['raga_id'],
                    'raga_name': match['raga_name'],
                    'confidence': match['confidence']
                })
        
        # Generate lo-fi if requested
        if not identify_only:
            # Only generate for the top raga
            if results['identified_ragas']:
                primary_raga = results['identified_ragas'][0]
                
                # Create generator
                generator = MelodicPatternGenerator()
                
                # Set the raga
                if generator.set_current_raga(primary_raga['raga_id']):
                    # Generate melody
                    generation_result = generator.generate_complete_track(
                        length=length,
                        creativity=creativity,
                        gamaka_intensity=gamaka_intensity
                    )
                    
                    if generation_result:
                        results['generated_files'] = [
                            generation_result['melody'],
                            generation_result['accompaniment']
                        ]
        
        results['success'] = True
        
    except Exception as e:
        results['error'] = str(e)
    
    return results


def batch_process(file_paths, output_dir="outputs", identify_only=False, preprocess=True,
                 creativity=0.5, gamaka_intensity=0.5, length=64, max_workers=4,
                 save_results=True):
    """
    Process multiple files in batch mode.
    
    Parameters:
    - file_paths: List of paths to audio files
    - output_dir: Directory for output files
    - identify_only: Whether to only identify the raga (no generation)
    - preprocess: Whether to preprocess the audio
    - creativity: Creativity level for melody generation (0.0-1.0)
    - gamaka_intensity: Intensity of gamakas in generation (0.0-1.0)
    - length: Length of melody to generate
    - max_workers: Maximum number of parallel workers
    - save_results: Whether to save results to a file
    
    Returns:
    - Dictionary with batch processing results
    """
    # Validate input files
    valid_files = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            valid_files.append(file_path)
        else:
            print(f"Warning: File not found - {file_path}")
    
    if not valid_files:
        print("No valid files to process.")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process files in parallel
    batch_results = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'files_processed': len(valid_files),
        'successful': 0,
        'failed': 0,
        'results': {}
    }
    
    print(f"Processing {len(valid_files)} files...")
    
    # Use ThreadPoolExecutor for parallelism
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(
                process_file, 
                file_path, 
                output_dir, 
                identify_only, 
                preprocess, 
                creativity, 
                gamaka_intensity, 
                length,
                save_results
            ): file_path for file_path in valid_files
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(valid_files)):
            file_path = future_to_file[future]
            file_name = os.path.basename(file_path)
            
            try:
                result = future.result()
                batch_results['results'][file_name] = result
                
                if result['success']:
                    batch_results['successful'] += 1
                else:
                    batch_results['failed'] += 1
                    print(f"Error processing {file_name}: {result['error']}")
                    
            except Exception as e:
                batch_results['results'][file_name] = {
                    'file_path': file_path,
                    'success': False,
                    'error': str(e)
                }
                batch_results['failed'] += 1
                print(f"Error processing {file_name}: {e}")
    
    # Save batch results
    if save_results:
        results_path = os.path.join(output_dir, f"batch_results_{int(time.time())}.json")
        with open(results_path, 'w') as f:
            json.dump(batch_results, f, indent=2)
        print(f"Batch results saved to: {results_path}")
    
    # Print summary
    print(f"\nBatch processing complete:")
    print(f"  Total files: {batch_results['files_processed']}")
    print(f"  Successful: {batch_results['successful']}")
    print(f"  Failed: {batch_results['failed']}")
    
    return batch_results


def process_from_csv(csv_file, output_dir="outputs", identify_only=False, preprocess=True,
                    creativity=0.5, gamaka_intensity=0.5, length=64, max_workers=4):
    """
    Process files listed in a CSV file.
    
    CSV format:
    file_path[,expected_raga_id]
    
    Parameters:
    - csv_file: Path to the CSV file
    - Other parameters same as batch_process()
    
    Returns:
    - Dictionary with batch processing results
    """
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found.")
        return None
    
    # Read the CSV file
    file_paths = []
    expected_ragas = {}
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue  # Skip empty rows and comments
                
            file_path = row[0].strip()
            if not file_path:
                continue
                
            file_paths.append(file_path)
            
            # If expected raga is provided
            if len(row) > 1 and row[1].strip():
                expected_ragas[file_path] = row[1].strip()
    
    # Process the files
    results = batch_process(
        file_paths,
        output_dir,
        identify_only,
        preprocess,
        creativity,
        gamaka_intensity,
        length,
        max_workers
    )
    
    if results and expected_ragas:
        # Calculate accuracy if expected ragas were provided
        correct = 0
        incorrect = 0
        
        for file_name, result in results['results'].items():
            file_path = result['file_path']
            if file_path in expected_ragas and result['success']:
                expected = expected_ragas[file_path]
                
                # Check if the top match matches the expected raga
                if (result['identified_ragas'] and 
                    result['identified_ragas'][0]['raga_id'] == expected):
                    correct += 1
                else:
                    incorrect += 1
        
        total_with_expected = correct + incorrect
        if total_with_expected > 0:
            accuracy = correct / total_with_expected
            results['accuracy'] = {
                'correct': correct,
                'incorrect': incorrect,
                'total_with_expected': total_with_expected,
                'accuracy': accuracy
            }
            
            print(f"\nAccuracy: {accuracy:.2%} ({correct}/{total_with_expected})")
    
    return results


def main():
    """Main function to run the batch processor from command line."""
    parser = argparse.ArgumentParser(
        description='Batch process audio files to identify ragas and generate lo-fi music')
    
    # File input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-d', '--directory', help='Process all files in directory')
    input_group.add_argument('-f', '--files', nargs='+', help='List of files to process')
    input_group.add_argument('-c', '--csv', help='CSV file with list of files to process')
    
    # Processing options
    parser.add_argument('-o', '--output-dir', default='outputs', 
                      help='Directory for output files')
    parser.add_argument('--identify-only', action='store_true',
                      help='Only identify ragas, do not generate music')
    parser.add_argument('--no-preprocess', action='store_false', dest='preprocess',
                      help='Disable audio preprocessing')
    
    # Generation options
    parser.add_argument('--creativity', type=float, default=0.5,
                      help='Creativity level for melody generation (0.0-1.0)')
    parser.add_argument('--gamaka', type=float, default=0.5,
                      help='Intensity of gamakas in generation (0.0-1.0)')
    parser.add_argument('--length', type=int, default=64,
                      help='Length of melody to generate')
    
    # Parallel processing
    parser.add_argument('--workers', type=int, default=4,
                      help='Maximum number of parallel workers')
    
    # File type filter
    parser.add_argument('--extensions', nargs='+', default=['.mp3', '.wav', '.flac', '.ogg'],
                      help='File extensions to process')
    
    args = parser.parse_args()
    
    # Collect files to process
    files_to_process = []
    
    if args.directory:
        # Process all matching files in the directory
        for root, _, files in os.walk(args.directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in args.extensions):
                    files_to_process.append(os.path.join(root, file))
        
        print(f"Found {len(files_to_process)} audio files in {args.directory}")
    
    elif args.files:
        # Process specified files
        files_to_process = args.files
    
    elif args.csv:
        # Process from CSV
        return process_from_csv(
            args.csv,
            args.output_dir,
            args.identify_only,
            args.preprocess,
            args.creativity,
            args.gamaka,
            args.length,
            args.workers
        )
    
    # Process the files
    batch_process(
        files_to_process,
        args.output_dir,
        args.identify_only,
        args.preprocess,
        args.creativity,
        args.gamaka,
        args.length,
        args.workers
    )


if __name__ == "__main__":
    main()