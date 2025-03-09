#!/usr/bin/env python3
"""
Raga Dataset Manager
-------------------
Handles labeled raga datasets and batch processing for model improvement.
"""

import os
import json
import csv
import shutil
import time
import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Import the analysis modules
from raga_identifier import RagaIdentifier
from raga_feature_extractor import RagaFeatureExtractor
from audio_analyzer import CarnaticAudioAnalyzer
from audio_preprocessor import CarnaticAudioPreprocessor

class RagaDatasetManager:
    """
    Manages labeled datasets of raga recordings and handles batch processing.
    """
    
    def __init__(self, dataset_dir='data/labeled_ragas', metadata_file='data/dataset_metadata.json'):
        """
        Initialize the dataset manager.
        
        Parameters:
        - dataset_dir: Directory for the labeled dataset
        - metadata_file: File to store dataset metadata
        """
        self.dataset_dir = dataset_dir
        self.metadata_file = metadata_file
        
        # Create directories if they don't exist
        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
        
        # Load or initialize metadata
        self.metadata = self._load_metadata()
        
        # Initialize analyzers
        self.identifier = RagaIdentifier()
        self.preprocessor = CarnaticAudioPreprocessor()
        
        # Track processing statistics
        self.stats = {
            'files_processed': 0,
            'files_skipped': 0,
            'errors': 0
        }
    
    def _load_metadata(self):
        """Load dataset metadata or create if it doesn't exist."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading metadata file. Creating new metadata.")
        
        # Initialize new metadata
        metadata = {
            'dataset_info': {
                'creation_date': datetime.datetime.now().isoformat(),
                'last_updated': datetime.datetime.now().isoformat(),
                'total_files': 0,
                'total_ragas': 0,
                'files_by_raga': {}
            },
            'files': {}
        }
        
        # Save the initialized metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def _save_metadata(self):
        """Save dataset metadata."""
        # Update last_updated timestamp
        self.metadata['dataset_info']['last_updated'] = datetime.datetime.now().isoformat()
        
        # Update counts
        self.metadata['dataset_info']['total_files'] = len(self.metadata['files'])
        
        # Count files by raga
        raga_counts = {}
        for file_info in self.metadata['files'].values():
            raga_id = file_info['raga_id']
            if raga_id not in raga_counts:
                raga_counts[raga_id] = 0
            raga_counts[raga_id] += 1
        
        self.metadata['dataset_info']['files_by_raga'] = raga_counts
        self.metadata['dataset_info']['total_ragas'] = len(raga_counts)
        
        # Save to file
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def add_file(self, file_path, raga_id, raga_name=None, artist=None, copy_file=True):
        """
        Add a labeled file to the dataset.
        
        Parameters:
        - file_path: Path to the audio file
        - raga_id: ID of the raga
        - raga_name: Name of the raga (optional)
        - artist: Artist name (optional)
        - copy_file: Whether to copy the file to the dataset directory
        
        Returns:
        - Path to the file in the dataset
        """
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found.")
            return None
        
        # Generate a unique ID for the file
        file_id = os.path.splitext(os.path.basename(file_path))[0]
        
        # If copying, create the raga subdirectory and copy the file
        dataset_path = file_path
        if copy_file:
            raga_dir = os.path.join(self.dataset_dir, raga_id)
            os.makedirs(raga_dir, exist_ok=True)
            
            # Check for file name conflicts
            base_name = os.path.basename(file_path)
            if os.path.exists(os.path.join(raga_dir, base_name)):
                # Add timestamp to make filename unique
                timestamp = int(time.time())
                base_name = f"{os.path.splitext(base_name)[0]}_{timestamp}{os.path.splitext(base_name)[1]}"
                
            # Copy the file
            dataset_path = os.path.join(raga_dir, base_name)
            shutil.copy2(file_path, dataset_path)
            print(f"Copied file to dataset: {dataset_path}")
        
        # Add to metadata
        self.metadata['files'][file_id] = {
            'file_path': dataset_path,
            'raga_id': raga_id,
            'raga_name': raga_name,
            'artist': artist,
            'added_date': datetime.datetime.now().isoformat(),
            'last_processed': None,
            'analysis_path': None
        }
        
        # Save updated metadata
        self._save_metadata()
        
        return dataset_path
    
    def import_from_csv(self, csv_file, copy_files=True):
        """
        Import multiple labeled files from a CSV file.
        
        CSV format:
        file_path,raga_id,raga_name,artist
        
        Parameters:
        - csv_file: Path to the CSV file
        - copy_files: Whether to copy files to the dataset directory
        
        Returns:
        - Number of files successfully imported
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file '{csv_file}' not found.")
            return 0
        
        # Read the CSV file
        imported = 0
        skipped = 0
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Check required fields
                if 'file_path' not in row or 'raga_id' not in row:
                    print(f"Error: Missing required fields in row: {row}")
                    skipped += 1
                    continue
                
                # Check if file exists
                if not os.path.exists(row['file_path']):
                    print(f"Error: File '{row['file_path']}' not found. Skipping.")
                    skipped += 1
                    continue
                
                # Add to dataset
                result = self.add_file(
                    row['file_path'],
                    row['raga_id'],
                    row.get('raga_name'),
                    row.get('artist'),
                    copy_files
                )
                
                if result:
                    imported += 1
                else:
                    skipped += 1
        
        print(f"Import complete: {imported} files imported, {skipped} files skipped")
        return imported
    
    def process_dataset(self, preprocess=True, update_models=True, force_reprocess=False):
        """
        Process all files in the dataset to extract features and update models.
        
        Parameters:
        - preprocess: Whether to preprocess the audio before analysis
        - update_models: Whether to update raga models with the results
        - force_reprocess: Whether to reprocess files that have already been processed
        
        Returns:
        - Dictionary with processing statistics
        """
        # Reset statistics
        self.stats = {
            'files_processed': 0,
            'files_skipped': 0,
            'errors': 0
        }
        
        # Process each file in the dataset
        for file_id, file_info in self.metadata['files'].items():
            # Skip if already processed and not forcing reprocess
            if file_info['last_processed'] and not force_reprocess:
                print(f"Skipping already processed file: {os.path.basename(file_info['file_path'])}")
                self.stats['files_skipped'] += 1
                continue
            
            # Check if file exists
            if not os.path.exists(file_info['file_path']):
                print(f"Error: File '{file_info['file_path']}' not found. Skipping.")
                self.stats['files_skipped'] += 1
                continue
            
            print(f"\nProcessing {os.path.basename(file_info['file_path'])} (Raga: {file_info['raga_id']})")
            
            try:
                # Set up output directory for analysis
                output_dir = os.path.join(self.dataset_dir, 'analysis', file_info['raga_id'])
                os.makedirs(output_dir, exist_ok=True)
                
                # Preprocess if requested
                processing_path = file_info['file_path']
                if preprocess:
                    try:
                        # Preprocess the audio
                        processed_path = os.path.join(
                            output_dir, 
                            f"{os.path.splitext(os.path.basename(file_info['file_path']))[0]}_processed.wav"
                        )
                        success = self.preprocessor.load_audio(file_info['file_path'])
                        if success:
                            self.preprocessor.process_for_analysis()
                            self.preprocessor.save_audio(processed_path)
                            processing_path = processed_path
                            print(f"Preprocessed audio saved to: {processed_path}")
                    except Exception as e:
                        print(f"Error preprocessing: {e}. Using original file.")
                
                # Analyze with the known raga ID
                results = self.identifier.analyze_file(
                    processing_path,
                    preprocess=False,  # Already preprocessed if requested
                    plot=False,
                    top_n=3
                )
                
                if results:
                    # Update with the correct raga information
                    self.provide_feedback(
                        results, 
                        file_info['raga_id'], 
                        file_info['raga_name'], 
                        update_models
                    )
                    
                    # Update metadata with processing info
                    self.metadata['files'][file_id]['last_processed'] = datetime.datetime.now().isoformat()
                    self.metadata['files'][file_id]['analysis_path'] = self.identifier.analysis_results.get('output_files', {}).get('full_results')
                    
                    self.stats['files_processed'] += 1
                else:
                    print(f"Error: Analysis failed for '{file_info['file_path']}'.")
                    self.stats['errors'] += 1
                
            except Exception as e:
                print(f"Error processing file: {e}")
                self.stats['errors'] += 1
        
        # Save updated metadata
        self._save_metadata()
        
        print(f"\nDataset processing complete:")
        print(f"  Files processed: {self.stats['files_processed']}")
        print(f"  Files skipped: {self.stats['files_skipped']}")
        print(f"  Errors: {self.stats['errors']}")
        
        return self.stats
    
    def provide_feedback(self, analysis_result, correct_raga_id, correct_raga_name=None, update_model=True):
        """
        Provide feedback about the correct raga for an analysis result.
        
        Parameters:
        - analysis_result: Analysis result dictionary
        - correct_raga_id: Correct raga ID
        - correct_raga_name: Correct raga name (optional)
        - update_model: Whether to update the raga model with this information
        
        Returns:
        - True if successful, False otherwise
        """
        if not analysis_result:
            print("No analysis result provided.")
            return False
        
        try:
            # Get the file path from the analysis
            file_path = analysis_result.get('file_path')
            if not file_path or not os.path.exists(file_path):
                print(f"Error: File path '{file_path}' not found in analysis or does not exist.")
                return False
            
            # Check if the identified raga matches the correct one
            top_match = None
            if ('overall_results' in analysis_result and 
                'top_matches' in analysis_result['overall_results'] and
                analysis_result['overall_results']['top_matches']):
                
                top_match = analysis_result['overall_results']['top_matches'][0]
                
                if top_match['raga_id'] == correct_raga_id:
                    print(f"Correct identification: {correct_raga_id} ({top_match['confidence']:.2f})")
                else:
                    print(f"Incorrect identification: {top_match['raga_id']} instead of {correct_raga_id}")
                    print(f"Confidence was: {top_match['confidence']:.2f}")
            else:
                print(f"No raga identified. Correct raga is: {correct_raga_id}")
            
            # If updating model, re-analyze with correct raga information
            if update_model:
                # Re-extract features
                feature_extractor = RagaFeatureExtractor()
                features = feature_extractor.analyze_file(file_path)
                
                if features:
                    # Update the model with correct raga ID
                    feature_extractor.update_raga_model(correct_raga_id, correct_raga_name)
                    feature_extractor.save_raga_models()
                    
                    print(f"Updated raga model for: {correct_raga_id}")
                    return True
                else:
                    print("Failed to extract features for model update.")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error providing feedback: {e}")
            return False
    
    def evaluate_accuracy(self):
        """
        Evaluate the system's accuracy on the labeled dataset.
        
        Returns:
        - Dictionary with accuracy metrics
        """
        # Initialize metrics
        metrics = {
            'total': 0,
            'correct': 0,
            'incorrect': 0,
            'top_3_correct': 0,
            'by_raga': {},
            'confusion_matrix': {}
        }
        
        # Create a temporary identifier for evaluation
        identifier = RagaIdentifier(segmentation=False)
        
        # Process each file in the dataset
        for file_id, file_info in self.metadata['files'].items():
            if not file_info['last_processed']:
                continue  # Skip unprocessed files
                
            # Check if file exists
            if not os.path.exists(file_info['file_path']):
                continue
            
            # Initialize raga-specific metrics if needed
            correct_raga = file_info['raga_id']
            if correct_raga not in metrics['by_raga']:
                metrics['by_raga'][correct_raga] = {
                    'total': 0,
                    'correct': 0,
                    'incorrect': 0
                }
            
            metrics['total'] += 1
            metrics['by_raga'][correct_raga]['total'] += 1
            
            # Analyze without updating model
            try:
                results = identifier.identify_raga(
                    file_info['file_path'],
                    preprocess=False,
                    plot=False,
                    top_n=3
                )
                
                if results and 'overall_results' in results and 'top_matches' in results['overall_results']:
                    top_matches = results['overall_results']['top_matches']
                    
                    if top_matches and top_matches[0]['raga_id'] == correct_raga:
                        # Correct top match
                        metrics['correct'] += 1
                        metrics['by_raga'][correct_raga]['correct'] += 1
                    else:
                        # Incorrect top match
                        metrics['incorrect'] += 1
                        metrics['by_raga'][correct_raga]['incorrect'] += 1
                        
                        # Record confusion
                        if top_matches:
                            predicted_raga = top_matches[0]['raga_id']
                            if correct_raga not in metrics['confusion_matrix']:
                                metrics['confusion_matrix'][correct_raga] = {}
                            
                            if predicted_raga not in metrics['confusion_matrix'][correct_raga]:
                                metrics['confusion_matrix'][correct_raga][predicted_raga] = 0
                            
                            metrics['confusion_matrix'][correct_raga][predicted_raga] += 1
                    
                    # Check if correct raga is in top 3
                    if any(match['raga_id'] == correct_raga for match in top_matches):
                        metrics['top_3_correct'] += 1
            
            except Exception as e:
                print(f"Error evaluating file {file_info['file_path']}: {e}")
        
        # Calculate accuracy percentages
        if metrics['total'] > 0:
            metrics['accuracy'] = metrics['correct'] / metrics['total']
            metrics['top_3_accuracy'] = metrics['top_3_correct'] / metrics['total']
            
            # Calculate per-raga accuracy
            for raga, raga_metrics in metrics['by_raga'].items():
                if raga_metrics['total'] > 0:
                    raga_metrics['accuracy'] = raga_metrics['correct'] / raga_metrics['total']
        else:
            metrics['accuracy'] = 0.0
            metrics['top_3_accuracy'] = 0.0
        
        return metrics
    
    def generate_report(self, output_file='dataset_report.md'):
        """
        Generate a markdown report about the dataset and model performance.
        
        Parameters:
        - output_file: Path to save the report
        
        Returns:
        - Path to the generated report
        """
        # Evaluate accuracy
        metrics = self.evaluate_accuracy()
        
        # Build the report
        report = [
            "# Raga Identification System Report",
            "",
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Dataset Summary",
            "",
            f"- Total Files: {self.metadata['dataset_info']['total_files']}",
            f"- Total Ragas: {self.metadata['dataset_info']['total_ragas']}",
            f"- Last Updated: {self.metadata['dataset_info']['last_updated']}",
            "",
            "### Files by Raga",
            ""
        ]
        
        # Add file counts by raga
        for raga, count in self.metadata['dataset_info']['files_by_raga'].items():
            report.append(f"- {raga}: {count} files")
        
        # Add accuracy metrics
        report.extend([
            "",
            "## Performance Metrics",
            "",
            f"- Total Files Evaluated: {metrics['total']}",
            f"- Top-1 Accuracy: {metrics['accuracy']:.2%}",
            f"- Top-3 Accuracy: {metrics['top_3_accuracy']:.2%}",
            "",
            "### Accuracy by Raga",
            ""
        ])
        
        # Add accuracy by raga
        for raga, raga_metrics in metrics['by_raga'].items():
            if raga_metrics['total'] > 0:
                report.append(f"- {raga}: {raga_metrics['accuracy']:.2%} ({raga_metrics['correct']}/{raga_metrics['total']})")
        
        # Add confusion matrix if there are errors
        if metrics['incorrect'] > 0 and metrics['confusion_matrix']:
            report.extend([
                "",
                "### Confusion Matrix",
                "",
                "| Actual \\ Predicted | " + " | ".join([raga for raga in metrics['by_raga'].keys()]) + " |",
                "|" + "-|" * (len(metrics['by_raga']) + 1)
            ])
            
            for actual in metrics['by_raga'].keys():
                row = [actual]
                for predicted in metrics['by_raga'].keys():
                    value = 0
                    if actual in metrics['confusion_matrix'] and predicted in metrics['confusion_matrix'][actual]:
                        value = metrics['confusion_matrix'][actual][predicted]
                    row.append(str(value))
                report.append("| " + " | ".join(row) + " |")
        
        # Add recommendations
        report.extend([
            "",
            "## Recommendations",
            ""
        ])
        
        # Add raga-specific recommendations
        low_accuracy_ragas = []
        for raga, raga_metrics in metrics['by_raga'].items():
            if raga_metrics['total'] >= 3 and raga_metrics['accuracy'] < 0.7:
                low_accuracy_ragas.append((raga, raga_metrics['accuracy']))
        
        if low_accuracy_ragas:
            report.append("### Ragas Needing More Samples:")
            for raga, accuracy in sorted(low_accuracy_ragas, key=lambda x: x[1]):
                report.append(f"- {raga}: Current accuracy {accuracy:.2%} is below target")
        
        # Add ragas with few samples
        few_samples = [(raga, count) for raga, count in self.metadata['dataset_info']['files_by_raga'].items() if count < 3]
        if few_samples:
            report.append("")
            report.append("### Ragas with Few Samples:")
            for raga, count in sorted(few_samples, key=lambda x: x[1]):
                report.append(f"- {raga}: Only {count} samples available")
        
        # Write the report
        with open(output_file, 'w') as f:
            f.write("\n".join(report))
        
        print(f"Report generated: {output_file}")
        return output_file


def create_sample_dataset():
    """Create a sample labeled dataset structure."""
    dataset_dir = 'data/labeled_ragas'
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create sample metadata file
    metadata = {
        'dataset_info': {
            'creation_date': datetime.datetime.now().isoformat(),
            'last_updated': datetime.datetime.now().isoformat(),
            'total_files': 0,
            'total_ragas': 0,
            'files_by_raga': {}
        },
        'files': {}
    }
    
    metadata_file = 'data/dataset_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create sample CSV file for import
    csv_content = """file_path,raga_id,raga_name,artist
path/to/your/yaman_recording.mp3,yaman,Yaman,Unknown
path/to/your/bhairav_recording.mp3,bhairav,Bhairav,Unknown
"""
    
    csv_file = os.path.join(dataset_dir, 'sample_import.csv')
    with open(csv_file, 'w') as f:
        f.write(csv_content)
    
    print(f"Created sample dataset structure at {dataset_dir}")
    print(f"Created sample import CSV at {csv_file}")
    print("Replace the file paths with your actual recording paths before importing")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage labeled raga datasets')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Create sample dataset command
    create_parser = subparsers.add_parser('create_sample', help='Create a sample dataset structure')
    
    # Import from CSV command
    import_parser = subparsers.add_parser('import', help='Import labeled files from CSV')
    import_parser.add_argument('csv_file', help='Path to the CSV file')
    import_parser.add_argument('--no-copy', action='store_false', dest='copy_files',
                             help='Do not copy files to the dataset directory')
    
    # Add single file command
    add_parser = subparsers.add_parser('add', help='Add a single labeled file')
    add_parser.add_argument('file', help='Path to the audio file')
    add_parser.add_argument('raga_id', help='ID of the raga')
    add_parser.add_argument('--name', help='Name of the raga')
    add_parser.add_argument('--artist', help='Artist name')
    add_parser.add_argument('--no-copy', action='store_false', dest='copy_file',
                          help='Do not copy file to the dataset directory')
    
    # Process dataset command
    process_parser = subparsers.add_parser('process', help='Process dataset and update models')
    process_parser.add_argument('--no-preprocess', action='store_false', dest='preprocess',
                              help='Do not preprocess audio')
    process_parser.add_argument('--no-update', action='store_false', dest='update_models',
                              help='Do not update raga models')
    process_parser.add_argument('--force', action='store_true', dest='force_reprocess',
                              help='Force reprocessing of already processed files')
    
    # Evaluate accuracy command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate system accuracy')
    eval_parser.add_argument('--report', help='Generate report file', default='dataset_report.md')
    
    args = parser.parse_args()
    
    if args.command == 'create_sample':
        create_sample_dataset()
    
    elif args.command == 'import':
        manager = RagaDatasetManager()
        manager.import_from_csv(args.csv_file, args.copy_files)
    
    elif args.command == 'add':
        manager = RagaDatasetManager()
        manager.add_file(args.file, args.raga_id, args.name, args.artist, args.copy_file)
    
    elif args.command == 'process':
        manager = RagaDatasetManager()
        manager.process_dataset(args.preprocess, args.update_models, args.force_reprocess)
    
    elif args.command == 'evaluate':
        manager = RagaDatasetManager()
        manager.generate_report(args.report)
    
    else:
        parser.print_help()