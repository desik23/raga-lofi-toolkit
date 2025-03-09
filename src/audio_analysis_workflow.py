#!/usr/bin/env python3
"""
Audio Analysis Workflow
----------------------
Integrates the various audio analysis modules into a complete workflow
for analyzing Carnatic music and generating features for lo-fi production.
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

# Import all analysis modules
from audio_analyzer import CarnaticAudioAnalyzer
from rhythm_pattern_analyzer import RhythmPatternAnalyzer
from raga_feature_extractor import RagaFeatureExtractor

# Optional - import generation modules if available
try:
    from melodic_pattern_generator import MelodicPatternGenerator
    GENERATION_AVAILABLE = True
except ImportError:
    GENERATION_AVAILABLE = False


class AudioAnalysisWorkflow:
    """
    Main class for orchestrating the complete audio analysis workflow.
    """
    
    def __init__(self, output_dir="outputs"):
        """
        Initialize the workflow.
        
        Parameters:
        - output_dir: Directory for output files
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize analyzers
        self.audio_analyzer = CarnaticAudioAnalyzer()
        self.rhythm_analyzer = RhythmPatternAnalyzer()
        self.raga_extractor = RagaFeatureExtractor(self.audio_analyzer)
        
        # Initialize pattern generator if available
        self.pattern_generator = None
        if GENERATION_AVAILABLE:
            self.pattern_generator = MelodicPatternGenerator()
    
    def analyze_file(self, file_path, raga_id=None, raga_name=None, plot=False):
        """
        Perform complete analysis on an audio file.
        
        Parameters:
        - file_path: Path to the audio file
        - raga_id: ID of the raga (if known)
        - raga_name: Name of the raga (if known)
        - plot: Whether to display plots during analysis
        
        Returns:
        - Dictionary with analysis results
        """
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found.")
            return None
        
        # Create results tracking dictionary
        results = {
            'file_path': file_path,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'raga_id': raga_id,
            'raga_name': raga_name,
            'analysis_modules': {},
            'output_files': {}
        }
        
        # Get filename without extension for output files
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Create run-specific output directory
        timestamp = int(time.time())
        run_dir = os.path.join(self.output_dir, f"{filename}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Step 1: Basic audio analysis
        print("\n1. Performing basic audio analysis...")
        try:
            success = self.audio_analyzer.load_audio(file_path)
            if not success:
                print("Error loading audio file. Aborting analysis.")
                return None
            
            self.audio_analyzer.extract_pitch()
            tonic = self.audio_analyzer.detect_tonic()
            note_events = self.audio_analyzer.extract_note_events()
            phrases = self.audio_analyzer.extract_phrases()
            
            # Save analysis results
            analysis_file = os.path.join(run_dir, f"{filename}_basic_analysis.pkl")
            self.audio_analyzer.save_analysis(analysis_file)
            
            # Update results
            results['analysis_modules']['basic_audio'] = {
                'tonic': self.audio_analyzer.tonic['note'] if self.audio_analyzer.tonic else None,
                'tonic_frequency': float(self.audio_analyzer.tonic['frequency']) if self.audio_analyzer.tonic else None,
                'note_events_count': len(note_events) if note_events else 0,
                'phrases_count': len(phrases) if phrases else 0
            }
            results['output_files']['basic_analysis'] = analysis_file
            
            print(f"  Detected tonic: {self.audio_analyzer.tonic['note'] if self.audio_analyzer.tonic else 'Unknown'}")
            print(f"  Extracted {len(note_events) if note_events else 0} note events")
            print(f"  Identified {len(phrases) if phrases else 0} phrases")
            
        except Exception as e:
            print(f"Error in basic audio analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
        # Step 2: Rhythm pattern analysis
        print("\n2. Analyzing rhythm patterns...")
        try:
            success = self.rhythm_analyzer.load_audio(file_path)
            if not success:
                print("Error loading audio file for rhythm analysis. Skipping rhythm analysis.")
            else:
                self.rhythm_analyzer.detect_onsets()
                self.rhythm_analyzer.estimate_tempo()
                rhythm_results = self.rhythm_analyzer.analyze_rhythm_patterns(plot=plot)
                
                # Save rhythm results
                rhythm_file = os.path.join(run_dir, f"{filename}_rhythm.json")
                self.rhythm_analyzer.export_rhythm_patterns(rhythm_file)
                
                # Update results
                results['analysis_modules']['rhythm'] = {
                    'tempo': float(rhythm_results['tempo']) if rhythm_results and 'tempo' in rhythm_results else None,
                    'tala': rhythm_results['detected_tala']['name'] if rhythm_results and 'detected_tala' in rhythm_results else 'Unknown',
                    'patterns_count': len(rhythm_results['rhythm_patterns']) if rhythm_results and 'rhythm_patterns' in rhythm_results else 0
                }
                results['output_files']['rhythm_analysis'] = rhythm_file
                
                if rhythm_results:
                    print(f"  Detected tempo: {rhythm_results['tempo']:.1f} BPM")
                    print(f"  Identified tala: {rhythm_results['detected_tala']['name']} (Confidence: {rhythm_results['detected_tala']['confidence']:.2f})")
                    print(f"  Extracted {len(rhythm_results['rhythm_patterns'])} rhythm patterns")
        except Exception as e:
            print(f"Error in rhythm analysis: {str(e)}")
            print("Continuing with other analysis steps...")
        
        # Step 3: Raga feature extraction
        print("\n3. Extracting raga features...")
        try:
            raga_features = self.raga_extractor.analyze_file(file_path)
            
            if raga_features:
                # Save raga features
                raga_file = os.path.join(run_dir, f"{filename}_raga_features.json")
                self.raga_extractor.export_raga_features_json(raga_file, raga_id, raga_name)
                
                # If raga_id is provided, update the model
                if raga_id:
                    self.raga_extractor.update_raga_model(raga_id, raga_name)
                    self.raga_extractor.save_raga_models()
                
                # Try to identify the raga if not specified
                if not raga_id and self.raga_extractor.raga_models:
                    identified_ragas = self.raga_extractor.identify_raga()
                    
                    if identified_ragas:
                        top_raga_id, confidence = identified_ragas[0]
                        
                        # Update results with identified raga
                        if 'raga_identification' not in results:
                            results['raga_identification'] = {}
                        
                        results['raga_identification']['top_match'] = top_raga_id
                        results['raga_identification']['confidence'] = float(confidence)
                        results['raga_identification']['alternatives'] = [
                            {'raga_id': raga_id, 'confidence': float(conf)}
                            for raga_id, conf in identified_ragas[1:3]
                        ]
                        
                        print(f"  Identified raga: {top_raga_id} (Confidence: {confidence:.2f})")
                        for raga_id, conf in identified_ragas[1:3]:
                            print(f"  Alternative match: {raga_id} (Confidence: {conf:.2f})")
                
                # Get key raga features
                arohana_avarohana = raga_features.get('arohana_avarohana', {})
                vadi_samvadi = raga_features.get('vadi_samvadi', {})
                gamaka_features = raga_features.get('gamaka_features', {})
                
                # Update results
                results['analysis_modules']['raga_features'] = {
                    'arohana': arohana_avarohana.get('arohana', []),
                    'avarohana': arohana_avarohana.get('avarohana', []),
                    'vadi': vadi_samvadi.get('vadi'),
                    'samvadi': vadi_samvadi.get('samvadi'),
                    'gamaka_percentage': gamaka_features.get('gamaka_percentage', 0),
                    'characteristic_phrases_count': len(raga_features.get('characteristic_phrases', []))
                }
                results['output_files']['raga_features'] = raga_file
                
                # Print some key features
                if arohana_avarohana:
                    indian_notes = ['Sa', 'r', 'R', 'g', 'G', 'm', 'M', 'P', 'd', 'D', 'n', 'N']
                    
                    arohana = arohana_avarohana.get('arohana', [])
                    avarohana = arohana_avarohana.get('avarohana', [])
                    
                    if arohana:
                        arohana_str = ' '.join(indian_notes[n] for n in arohana)
                        print(f"  Arohana (ascending): {arohana_str}")
                    
                    if avarohana:
                        avarohana_str = ' '.join(indian_notes[n] for n in avarohana)
                        print(f"  Avarohana (descending): {avarohana_str}")
                
                if vadi_samvadi and 'vadi' in vadi_samvadi and vadi_samvadi['vadi'] is not None:
                    indian_notes = ['Sa', 'r', 'R', 'g', 'G', 'm', 'M', 'P', 'd', 'D', 'n', 'N']
                    print(f"  Vadi (dominant note): {indian_notes[vadi_samvadi['vadi']]}")
                    if 'samvadi' in vadi_samvadi and vadi_samvadi['samvadi'] is not None:
                        print(f"  Samvadi (second dominant): {indian_notes[vadi_samvadi['samvadi']]}")
                
                # Plot raga features if requested
                if plot:
                    self.raga_extractor.plot_raga_features()
                
            else:
                print("  No raga features extracted.")
                
        except Exception as e:
            print(f"Error in raga feature extraction: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Step 4: Generate example patterns if generator is available
        if self.pattern_generator:
            print("\n4. Generating example patterns...")
            try:
                # If raga was identified or provided, generate patterns
                use_raga_id = raga_id
                if not use_raga_id and 'raga_identification' in results and 'top_match' in results['raga_identification']:
                    use_raga_id = results['raga_identification']['top_match']
                
                if use_raga_id and self.pattern_generator.set_current_raga(use_raga_id):
                    # Generate 16-note phrase as example
                    example_phrase = self.pattern_generator.generate_melodic_phrase(16, creativity=0.5)
                    
                    if example_phrase:
                        # Convert to Indian notation
                        indian_notes = ['Sa', 'r', 'R', 'g', 'G', 'm', 'M', 'P', 'd', 'D', 'n', 'N']
                        phrase_str = ' '.join(indian_notes[n % 12] for n in example_phrase)
                        
                        # Save as MIDI
                        example_midi = os.path.join(run_dir, f"{filename}_example_phrase.mid")
                        self.pattern_generator.create_midi_sequence(example_phrase, example_midi)
                        
                        # Update results
                        results['example_patterns'] = {
                            'phrase': example_phrase,
                            'notation': phrase_str
                        }
                        results['output_files']['example_midi'] = example_midi
                        
                        print(f"  Generated example phrase: {phrase_str}")
                        print(f"  Saved example MIDI to: {example_midi}")
                else:
                    print("  No suitable raga model found for pattern generation.")
            except Exception as e:
                print(f"Error in pattern generation: {str(e)}")
        
        # Save complete analysis results
        results_file = os.path.join(run_dir, f"{filename}_analysis_results.json")
        try:
            with open(results_file, 'w') as f:
                # Convert NumPy types to Python native types for JSON serialization
                def convert_for_json(obj):
                    if hasattr(obj, 'item'):  # NumPy scalar
                        return obj.item()
                    elif hasattr(obj, 'tolist'):  # NumPy array
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_for_json(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_for_json(i) for i in obj]
                    else:
                        return obj
                
                json.dump(convert_for_json(results), f, indent=2)
            
            results['output_files']['full_results'] = results_file
            print(f"\nComplete analysis results saved to: {results_file}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")
        
        return results

    def analyze_directory(self, directory, raga_id=None, raga_name=None, recursive=False, file_types=None):
        """
        Analyze all audio files in a directory.
        
        Parameters:
        - directory: Directory containing audio files
        - raga_id: ID of the raga (if known and same for all files)
        - raga_name: Name of the raga (if known and same for all files)
        - recursive: Whether to recursively process subdirectories
        - file_types: List of file extensions to process (default: ['.wav', '.mp3', '.ogg', '.flac'])
        
        Returns:
        - Dictionary with analysis results for each file
        """
        if not os.path.isdir(directory):
            print(f"Error: '{directory}' is not a valid directory.")
            return None
        
        # Default file types
        if file_types is None:
            file_types = ['.wav', '.mp3', '.ogg', '.flac']
        
        # Convert to lowercase for case-insensitive matching
        file_types = [ext.lower() for ext in file_types]
        
        # Find all matching files
        audio_files = []
        if recursive:
            for root, _, files in os.walk(directory):
                for file in files:
                    if os.path.splitext(file)[1].lower() in file_types:
                        audio_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory):
                if os.path.splitext(file)[1].lower() in file_types:
                    audio_files.append(os.path.join(directory, file))
        
        if not audio_files:
            print(f"No audio files found in '{directory}'.")
            return None
        
        print(f"Found {len(audio_files)} audio files to analyze.")
        
        # Process each file
        results = {}
        for i, file_path in enumerate(audio_files):
            print(f"\nProcessing file {i+1}/{len(audio_files)}: {os.path.basename(file_path)}")
            file_result = self.analyze_file(file_path, raga_id, raga_name, plot=False)
            if file_result:
                results[file_path] = file_result
        
        # Save summary of all results
        summary_file = os.path.join(self.output_dir, f"analysis_summary_{int(time.time())}.json")
        try:
            with open(summary_file, 'w') as f:
                # Simplify results for summary
                summary = {
                    'directory': directory,
                    'raga_id': raga_id,
                    'raga_name': raga_name,
                    'files_processed': len(audio_files),
                    'files_analyzed': len(results),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'file_summaries': {}
                }
                
                for file_path, file_result in results.items():
                    filename = os.path.basename(file_path)
                    summary['file_summaries'][filename] = {
                        'identified_raga': file_result.get('raga_identification', {}).get('top_match', raga_id),
                        'tonic': file_result.get('analysis_modules', {}).get('basic_audio', {}).get('tonic'),
                        'tempo': file_result.get('analysis_modules', {}).get('rhythm', {}).get('tempo'),
                        'output_directory': os.path.dirname(file_result.get('output_files', {}).get('full_results', ''))
                    }
                
                json.dump(summary, f, indent=2)
            
            print(f"\nAnalysis summary saved to: {summary_file}")
        except Exception as e:
            print(f"Error saving summary: {str(e)}")
        
        return results


def main():
    """Main function to run the workflow from command line."""
    parser = argparse.ArgumentParser(description='Carnatic Music Audio Analysis Workflow')
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-f', '--file', help='Path to audio file for analysis')
    input_group.add_argument('-d', '--directory', help='Directory containing audio files to analyze')
    
    # Raga information
    parser.add_argument('-r', '--raga-id', help='ID of the raga (if known)')
    parser.add_argument('-n', '--raga-name', help='Name of the raga (if known)')
    
    # Output options
    parser.add_argument('-o', '--output-dir', default='outputs', help='Directory for output files')
    parser.add_argument('-p', '--plot', action='store_true', help='Display plots during analysis')
    
    # Directory processing options
    parser.add_argument('--recursive', action='store_true', help='Recursively process subdirectories')
    parser.add_argument('--file-types', nargs='+', default=['.wav', '.mp3', '.ogg', '.flac'], 
                       help='File extensions to process (default: .wav .mp3 .ogg .flac)')
    
    args = parser.parse_args()
    
    # Create workflow
    workflow = AudioAnalysisWorkflow(output_dir=args.output_dir)
    
    # Process input
    if args.file:
        print(f"\nAnalyzing file: {args.file}")
        workflow.analyze_file(args.file, args.raga_id, args.raga_name, plot=args.plot)
    else:
        print(f"\nAnalyzing directory: {args.directory}")
        workflow.analyze_directory(args.directory, args.raga_id, args.raga_name, 
                                 recursive=args.recursive, file_types=args.file_types)
    
    print("\nAnalysis workflow completed.")


if __name__ == "__main__":
    main()