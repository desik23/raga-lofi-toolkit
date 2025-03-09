#!/usr/bin/env python3
"""
Raga Identifier
--------------
Analyzes Carnatic and Hindustani music recordings to identify the raga.
Handles mixed ragas, creative interpretations, and provides multiple
likely matches with confidence scores.
"""

import os
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from datetime import datetime
import soundfile as sf

# Import analysis modules
from audio_analyzer import CarnaticAudioAnalyzer
from raga_feature_extractor import RagaFeatureExtractor
from audio_preprocessor import CarnaticAudioPreprocessor

class RagaIdentifier:
    """
    Specialized class for identifying ragas in music recordings,
    handling mixed ragas and creative interpretations.
    """
    
    def __init__(self, models_dir='data/raga_models', segmentation=True):
        """
        Initialize the raga identifier.
        
        Parameters:
        - models_dir: Directory containing raga models
        - segmentation: Whether to analyze segments separately for mixed ragas
        """
        self.models_dir = models_dir
        self.use_segmentation = segmentation
        
        # Initialize analyzers
        self.preprocessor = CarnaticAudioPreprocessor()
        self.audio_analyzer = CarnaticAudioAnalyzer()
        self.feature_extractor = RagaFeatureExtractor(self.audio_analyzer)
        
        # Load raga metadata
        self.raga_metadata = self._load_raga_metadata()
        
        # Dictionary to store analysis results
        self.analysis_results = {}
        self.segment_results = []
        
        # Load raga models
        self._load_raga_models()
    
    def _load_raga_metadata(self, metadata_file='data/raga_metadata.json'):
        """Load additional metadata about ragas."""
        metadata = {}
        try:
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                print(f"Raga metadata file not found: {metadata_file}")
                # Create minimal metadata structure
                metadata = {
                    "categories": {
                        "time_of_day": {
                            "morning": [],
                            "afternoon": [],
                            "evening": [],
                            "night": []
                        },
                        "moods": {
                            "peaceful": [],
                            "devotional": [],
                            "romantic": [],
                            "heroic": [],
                            "melancholic": []
                        }
                    },
                    "similar_ragas": {}
                }
        except Exception as e:
            print(f"Error loading raga metadata: {e}")
        
        return metadata
    
    def _load_raga_models(self):
        """Load all available raga models for identification."""
        # First try to load from pickle file
        model_file = os.path.join(self.models_dir, 'raga_models.pkl')
        if os.path.exists(model_file):
            try:
                with open(model_file, 'rb') as f:
                    self.feature_extractor.raga_models = pickle.load(f)
                print(f"Loaded {len(self.feature_extractor.raga_models)} raga models from {model_file}")
                return
            except Exception as e:
                print(f"Error loading raga models from pickle: {e}")
        
        # If pickle fails or doesn't exist, load individual JSON files
        if not os.path.exists(self.models_dir):
            print(f"Models directory not found: {self.models_dir}")
            return
        
        loaded_models = 0
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.json'):
                try:
                    file_path = os.path.join(self.models_dir, filename)
                    with open(file_path, 'r') as f:
                        raga_data = json.load(f)
                    
                    # Extract raga ID from metadata or filename
                    raga_id = None
                    if 'metadata' in raga_data and 'raga_id' in raga_data['metadata']:
                        raga_id = raga_data['metadata']['raga_id']
                    else:
                        # Try to extract from filename
                        raga_id = filename.split('_')[0]
                    
                    if raga_id:
                        self.feature_extractor.raga_models[raga_id] = raga_data
                        loaded_models += 1
                except Exception as e:
                    print(f"Error loading raga file {filename}: {e}")
        
        print(f"Loaded {loaded_models} raga models from individual files")
    
    def identify_raga(self, file_path, preprocess=True, top_n=3, plot=False):
        """
        Identify the raga(s) in an audio file.
        
        Parameters:
        - file_path: Path to the audio file
        - preprocess: Whether to preprocess the audio
        - top_n: Number of top matches to return
        - plot: Whether to display plots
        
        Returns:
        - Dictionary with identification results
        """
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found.")
            return None
        
        print(f"Analyzing file: {file_path}")
        
        # Reset results
        self.analysis_results = {
            'file_path': file_path,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'overall_results': {},
            'segment_results': [],
            'confidence': 0.0
        }
        
        # Preprocess the audio if requested
        if preprocess:
            print("Preprocessing audio...")
            processed_path = self._preprocess_audio(file_path)
            if processed_path:
                file_path = processed_path
        
        # Check if we should use segmentation
        if self.use_segmentation:
            # Analyze segments separately for potential mixed ragas
            return self._analyze_with_segments(file_path, top_n, plot)
        else:
            # Analyze the whole file as a single raga
            return self._analyze_full_file(file_path, top_n, plot)
    
    def _preprocess_audio(self, file_path):
        """Preprocess the audio for better analysis."""
        try:
            # Load the audio
            success = self.preprocessor.load_audio(file_path)
            if not success:
                print("Error loading audio for preprocessing. Using original file.")
                return None
            
            # Apply standard processing steps
            self.preprocessor.process_for_analysis(
                normalize_tonic=True,
                target_tonic='C4',
                remove_silence=True,
                apply_bandpass=True,
                normalize_volume=True
            )
            
            # Generate output path
            base_dir = os.path.dirname(file_path)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            processed_path = os.path.join(base_dir, f"{base_name}_processed_for_id.wav")
            
            # Save processed audio
            return self.preprocessor.save_audio(processed_path)
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None
    
    def _analyze_full_file(self, file_path, top_n=3, plot=False):
        """Analyze the whole file as a single raga."""
        # Extract features
        features = self.feature_extractor.analyze_file(file_path)
        
        if not features:
            print("Failed to extract features.")
            return None
        
        # Initialize analysis results structure
        self.analysis_results = {
            'file_path': file_path,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'overall_results': {'top_matches': []}  # Initialize with empty list
        }
        
        # Identify raga
        raga_matches = self.feature_extractor.identify_raga(top_n=top_n)
        
        # Store results
        top_matches = []
        for raga_id, confidence in raga_matches:
            raga_name = "Unknown"
            for model_id, model in self.feature_extractor.raga_models.items():
                if model_id == raga_id:
                    if 'metadata' in model and 'raga_name' in model['metadata']:
                        raga_name = model['metadata']['raga_name']
                    elif 'name' in model:
                        raga_name = model['name']
                    break
            
            # Find similar ragas
            similar_ragas = self._find_similar_ragas(raga_id)
            
            top_matches.append({
                'raga_id': raga_id,
                'raga_name': raga_name,
                'confidence': float(confidence),
                'similar_ragas': similar_ragas
            })
        
        # Update analysis results with top matches
        self.analysis_results['overall_results']['top_matches'] = top_matches
        
        if top_matches:
            self.analysis_results['overall_results']['single_raga_confidence'] = float(top_matches[0]['confidence'])
            
            # Extract key features for the top match
            top_raga = top_matches[0]['raga_id']
            features_summary = self._extract_features_summary(features)
            top_matches[0]['features'] = features_summary
        else:
            self.analysis_results['overall_results']['single_raga_confidence'] = 0.0
            
        # Plot if requested
        if plot and top_matches:
            self._plot_raga_analysis(features, top_matches)
        
        return self.analysis_results
    
    def _analyze_with_segments(self, file_path, top_n=3, plot=False):
        """Analyze segments separately for potential mixed ragas."""
        # First analyze the full file
        full_file_results = self._analyze_full_file(file_path, top_n, False)  # No plot yet
        
        if not full_file_results:
            return None
        
        # Segment the audio
        print("Segmenting audio for mixed raga analysis...")
        try:
            success = self.preprocessor.load_audio(file_path)
            if not success:
                print("Error loading audio for segmentation.")
                return full_file_results
            
            # Use two segmentation approaches
            segments1 = self.preprocessor.segment_audio(segment_length=60, with_overlap=False)
            segments2 = self.preprocessor.extract_main_sections(section_count=3)
            
            # Combine segments from both approaches
            all_segments = []
            
            # Add segments from first approach
            if segments1:
                for i, segment in enumerate(segments1):
                    all_segments.append((segment, 0, len(segment) / self.preprocessor.sample_rate, f"Segment {i+1}"))
            
            # Add segments from second approach
            if segments2:
                all_segments.extend(segments2)
            
            # Sort by start time
            all_segments.sort(key=lambda x: x[1])
            
            # Analyze each segment
            segment_results = []
            
            for i, (segment, start_time, end_time, label) in enumerate(all_segments):
                print(f"\nAnalyzing {label} ({start_time:.2f}s - {end_time:.2f}s)...")
                
                # Save segment audio
                segment_path = os.path.join(os.path.dirname(file_path), f"temp_segment_{i}.wav")
                sf.write(segment_path, segment, self.preprocessor.sample_rate)
                
                try:
                    # Extract features
                    segment_features = self.feature_extractor.analyze_file(segment_path)
                    
                    if segment_features:
                        # Identify raga for this segment
                        segment_matches = self.feature_extractor.identify_raga(top_n=top_n)
                        
                        if segment_matches:
                            segment_top_matches = []
                            for raga_id, confidence in segment_matches:
                                raga_name = "Unknown"
                                for model_id, model in self.feature_extractor.raga_models.items():
                                    if model_id == raga_id:
                                        if 'metadata' in model and 'raga_name' in model['metadata']:
                                            raga_name = model['metadata']['raga_name']
                                        elif 'name' in model:
                                            raga_name = model['name']
                                        break
                                
                                segment_top_matches.append({
                                    'raga_id': raga_id,
                                    'raga_name': raga_name,
                                    'confidence': float(confidence)
                                })
                            
                            # Store segment results
                            segment_results.append({
                                'segment_label': label,
                                'start_time': float(start_time),
                                'end_time': float(end_time),
                                'top_matches': segment_top_matches
                            })
                except Exception as e:
                    print(f"Error analyzing segment: {e}")
                
                # Clean up temp file
                try:
                    os.remove(segment_path)
                except:
                    pass
            
            # Analyze segment results to detect mixed ragas
            mixed_raga_analysis = self._analyze_mixed_ragas(segment_results)
            
            # Update the analysis results
            self.analysis_results['segment_results'] = segment_results
            self.analysis_results['mixed_raga_analysis'] = mixed_raga_analysis
            
            # Determine if it's likely a mixed raga composition
            if mixed_raga_analysis['is_mixed_raga']:
                self.analysis_results['overall_results']['type'] = 'mixed_raga'
                self.analysis_results['overall_results']['mixed_raga_confidence'] = mixed_raga_analysis['confidence']
            else:
                self.analysis_results['overall_results']['type'] = 'single_raga'
            
            # Plot if requested
            if plot:
                self._plot_segment_analysis(segment_results, mixed_raga_analysis)
            
            return self.analysis_results
        
        except Exception as e:
            print(f"Error in segment analysis: {e}")
            # Fall back to full file analysis
            return full_file_results
    
    def _analyze_mixed_ragas(self, segment_results):
        """
        Analyze segment results to detect mixed ragas.
        
        Parameters:
        - segment_results: List of results for each analyzed segment
        
        Returns:
        - Dictionary with mixed raga analysis
        """
        if not segment_results:
            return {
                'is_mixed_raga': False,
                'confidence': 0.0,
                'transitions': []
            }
        
        # Count unique ragas across segments
        raga_counts = defaultdict(int)
        segment_ragas = []
        
        for segment in segment_results:
            if 'top_matches' in segment and segment['top_matches']:
                top_raga = segment['top_matches'][0]['raga_id']
                segment_ragas.append((segment['segment_label'], top_raga, segment['top_matches'][0]['confidence']))
                raga_counts[top_raga] += 1
        
        # Determine if this is likely a mixed raga composition
        unique_ragas = len(raga_counts)
        
        # Find transitions between different ragas
        transitions = []
        prev_raga = None
        
        for i, (label, raga, confidence) in enumerate(segment_ragas):
            if prev_raga and raga != prev_raga:
                transitions.append({
                    'from': prev_raga,
                    'to': raga,
                    'segment': label,
                    'confidence': confidence
                })
            prev_raga = raga
        
        # Calculate confidence in mixed raga determination
        # Higher if there are clear transitions and unique ragas
        mixed_confidence = 0.0
        
        if unique_ragas > 1:
            # Basic confidence based on having multiple ragas
            mixed_confidence = min(0.5 + (unique_ragas - 1) * 0.15, 0.95)
            
            # Adjust based on transition clarity
            if transitions:
                transition_confidences = [t['confidence'] for t in transitions]
                avg_transition_confidence = sum(transition_confidences) / len(transition_confidences)
                mixed_confidence *= (0.5 + 0.5 * avg_transition_confidence)
        
        is_mixed = mixed_confidence > 0.4  # Threshold for mixed raga determination
        
        return {
            'is_mixed_raga': is_mixed,
            'confidence': mixed_confidence,
            'unique_ragas': unique_ragas,
            'transitions': transitions
        }
    
    def _extract_features_summary(self, features):
        """Extract a summary of key features for display."""
        summary = {}
        
        # Extract arohana/avarohana
        arohana_avarohana = features.get('arohana_avarohana', {})
        if arohana_avarohana:
            indian_notes = ['Sa', 'r', 'R', 'g', 'G', 'm', 'M', 'P', 'd', 'D', 'n', 'N']
            
            arohana = arohana_avarohana.get('arohana', [])
            avarohana = arohana_avarohana.get('avarohana', [])
            
            if arohana:
                summary['arohana'] = ' '.join(indian_notes[n % 12] for n in arohana)
            
            if avarohana:
                summary['avarohana'] = ' '.join(indian_notes[n % 12] for n in avarohana)
        
        # Extract vadi/samvadi
        vadi_samvadi = features.get('vadi_samvadi', {})
        if vadi_samvadi:
            indian_notes = ['Sa', 'r', 'R', 'g', 'G', 'm', 'M', 'P', 'd', 'D', 'n', 'N']
            
            if 'vadi' in vadi_samvadi and vadi_samvadi['vadi'] is not None:
                summary['vadi'] = indian_notes[vadi_samvadi['vadi'] % 12]
            
            if 'samvadi' in vadi_samvadi and vadi_samvadi['samvadi'] is not None:
                summary['samvadi'] = indian_notes[vadi_samvadi['samvadi'] % 12]
        
        return summary
    
    def _find_similar_ragas(self, raga_id):
        """Find ragas similar to the identified raga."""
        similar_ragas = []
        
        # Check in metadata
        if (self.raga_metadata and 'similar_ragas' in self.raga_metadata and 
            raga_id in self.raga_metadata['similar_ragas']):
            return self.raga_metadata['similar_ragas'][raga_id]
        
        # If not in metadata, use feature comparison
        if not self.feature_extractor.raga_models:
            return similar_ragas
        
        # Get the raga model
        raga_model = None
        for model_id, model in self.feature_extractor.raga_models.items():
            if model_id == raga_id:
                raga_model = model
                break
        
        if not raga_model:
            return similar_ragas
        
        # Compare to other ragas
        similarities = []
        
        for other_id, other_model in self.feature_extractor.raga_models.items():
            if other_id == raga_id:
                continue
            
            # Calculate similarity based on scale
            similarity = self._calculate_raga_similarity(raga_model, other_model)
            
            if similarity > 0.6:  # Minimum similarity threshold
                raga_name = "Unknown"
                if 'metadata' in other_model and 'raga_name' in other_model['metadata']:
                    raga_name = other_model['metadata']['raga_name']
                elif 'name' in other_model:
                    raga_name = other_model['name']
                
                similarities.append((other_id, raga_name, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        # Return top 3 similar ragas
        for raga_id, raga_name, similarity in similarities[:3]:
            similar_ragas.append({
                'raga_id': raga_id,
                'raga_name': raga_name,
                'similarity': float(similarity)
            })
        
        return similar_ragas
    
    def _calculate_raga_similarity(self, raga1, raga2):
        """Calculate similarity between two ragas based on features."""
        similarity = 0.0
        
        # Compare arohana/avarohana
        if ('arohana_avarohana' in raga1 and 'arohana_avarohana' in raga2):
            arohana1 = set(raga1['arohana_avarohana'].get('arohana', []))
            arohana2 = set(raga2['arohana_avarohana'].get('arohana', []))
            
            avarohana1 = set(raga1['arohana_avarohana'].get('avarohana', []))
            avarohana2 = set(raga2['arohana_avarohana'].get('avarohana', []))
            
            # Jaccard similarity for arohana/avarohana
            if arohana1 and arohana2:
                arohana_sim = len(arohana1.intersection(arohana2)) / len(arohana1.union(arohana2))
            else:
                arohana_sim = 0.0
                
            if avarohana1 and avarohana2:
                avarohana_sim = len(avarohana1.intersection(avarohana2)) / len(avarohana1.union(avarohana2))
            else:
                avarohana_sim = 0.0
            
            # Weight more for scale similarity
            similarity += 0.6 * (arohana_sim + avarohana_sim) / 2
        
        # Compare vadi/samvadi
        if ('vadi_samvadi' in raga1 and 'vadi_samvadi' in raga2):
            vadi1 = raga1['vadi_samvadi'].get('vadi')
            vadi2 = raga2['vadi_samvadi'].get('vadi')
            
            samvadi1 = raga1['vadi_samvadi'].get('samvadi')
            samvadi2 = raga2['vadi_samvadi'].get('samvadi')
            
            # Check if vadi/samvadi match
            vadi_match = (vadi1 is not None and vadi2 is not None and vadi1 == vadi2)
            samvadi_match = (samvadi1 is not None and samvadi2 is not None and samvadi1 == samvadi2)
            
            if vadi_match:
                similarity += 0.2
            if samvadi_match:
                similarity += 0.1
        
        # Compare time of day
        time1 = None
        time2 = None
        
        if self.raga_metadata and 'categories' in self.raga_metadata:
            for time_period, ragas in self.raga_metadata['categories']['time_of_day'].items():
                if 'metadata' in raga1 and 'raga_id' in raga1['metadata'] and raga1['metadata']['raga_id'] in ragas:
                    time1 = time_period
                if 'metadata' in raga2 and 'raga_id' in raga2['metadata'] and raga2['metadata']['raga_id'] in ragas:
                    time2 = time_period
        
        if time1 and time2 and time1 == time2:
            similarity += 0.1
        
        return similarity
    
    def _plot_raga_analysis(self, features, top_matches):
        """Plot visualizations of the raga analysis."""
        if not features:
            return
        
        plt.figure(figsize=(15, 10))
        
        # 1. Plot note distribution
        plt.subplot(2, 2, 1)
        self._plot_note_distribution(features)
        
        # 2. Plot top matches
        plt.subplot(2, 2, 2)
        self._plot_top_matches(top_matches)
        
        # 3. Plot transition matrix
        plt.subplot(2, 2, 3)
        self._plot_transition_matrix(features)
        
        # 4. Plot arohana/avarohana
        plt.subplot(2, 2, 4)
        self._plot_arohana_avarohana(features)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_note_distribution(self, features):
        """Plot the note distribution."""
        note_dist = features.get('note_distribution', {})
        
        if not note_dist or 'note_percentages' not in note_dist:
            plt.title("Note Distribution (No Data)")
            return
        
        percentages = note_dist['note_percentages']
        
        # Convert to ordered list
        notes = []
        values = []
        
        # Use Indian note names
        indian_notes = ['Sa', 'r', 'R', 'g', 'G', 'm', 'M', 'P', 'd', 'D', 'n', 'N']
        
        for note_str, percentage in percentages.items():
            note = int(note_str) % 12
            notes.append(indian_notes[note])
            values.append(percentage)
        
        # Plot
        plt.bar(notes, values)
        plt.title("Note Distribution")
        plt.xlabel("Note")
        plt.ylabel("Percentage (%)")
        plt.xticks(rotation=45)
        
        # Highlight vadi and samvadi
        vadi_samvadi = features.get('vadi_samvadi', {})
        
        if 'vadi' in vadi_samvadi and vadi_samvadi['vadi'] is not None:
            vadi = vadi_samvadi['vadi'] % 12
            note_key = str(vadi)
            if note_key in percentages:
                plt.bar(indian_notes[vadi], percentages[note_key], color='red', label=f'Vadi: {indian_notes[vadi]}')
        
        if 'samvadi' in vadi_samvadi and vadi_samvadi['samvadi'] is not None:
            samvadi = vadi_samvadi['samvadi'] % 12
            note_key = str(samvadi)
            if note_key in percentages:
                plt.bar(indian_notes[samvadi], percentages[note_key], color='green', label=f'Samvadi: {indian_notes[samvadi]}')
        
        plt.legend()
    
    def _plot_top_matches(self, top_matches):
        """Plot the top raga matches."""
        if not top_matches:
            plt.title("Top Matches (No Data)")
            return
        
        # Extract data
        labels = [match['raga_name'] for match in top_matches]
        values = [match['confidence'] for match in top_matches]
        
        # Plot as horizontal bar chart
        plt.barh(labels, values, color='skyblue')
        plt.title("Top Raga Matches")
        plt.xlabel("Confidence Score")
        plt.xlim(0, 1)
        
        # Add confidence values
        for i, v in enumerate(values):
            plt.text(v + 0.01, i, f"{v:.2f}", va='center')
    
    def _plot_transition_matrix(self, features):
        """Plot the transition matrix as a heatmap."""
        transition_data = features.get('transition_matrix', {})
        
        if not transition_data or 'matrix' not in transition_data:
            plt.title("Transition Matrix (No Data)")
            return
        
        matrix_dict = transition_data['matrix']
        
        # Convert dictionary to matrix
        matrix = np.zeros((12, 12))
        
        for from_note_str, transitions in matrix_dict.items():
            from_note = int(from_note_str)
            for to_note_str, prob in transitions.items():
                to_note = int(to_note_str)
                matrix[from_note % 12, to_note % 12] = prob
        
        # Plot heatmap
        plt.imshow(matrix, cmap='Blues')
        plt.colorbar(label='Transition Probability')
        plt.title("Note Transition Matrix")
        
        # Use Indian note names for labels
        indian_notes = ['Sa', 'r', 'R', 'g', 'G', 'm', 'M', 'P', 'd', 'D', 'n', 'N']
        
        plt.xticks(range(12), indian_notes)
        plt.yticks(range(12), indian_notes)
        plt.xlabel("To Note")
        plt.ylabel("From Note")
    
    def _plot_arohana_avarohana(self, features):
        """Plot arohana and avarohana patterns."""
        arohana_avarohana = features.get('arohana_avarohana', {})
        
        if not arohana_avarohana:
            plt.title("Arohana/Avarohana (No Data)")
            return
        
        arohana = arohana_avarohana.get('arohana', [])
        avarohana = arohana_avarohana.get('avarohana', [])
        
        # Plot as a grid showing which notes are present
        grid = np.zeros((2, 12))
        
        # Fill in the grid
        for note in arohana:
            grid[0, note % 12] = 1
            
        for note in avarohana:
            grid[1, note % 12] = 1
        
        # Plot
        plt.imshow(grid, cmap='Blues', aspect='auto')
        
        # Use Indian note names for labels
        indian_notes = ['Sa', 'r', 'R', 'g', 'G', 'm', 'M', 'P', 'd', 'D', 'n', 'N']
        plt.xticks(range(12), indian_notes)
        plt.yticks([0, 1], ['Arohana', 'Avarohana'])
        plt.title("Scale Structure")
        
        # Add annotations
        for i in range(12):
            for j in range(2):
                if grid[j, i] == 1:
                    plt.text(i, j, '•', ha='center', va='center', color='white', fontsize=15)
    
    def _plot_segment_analysis(self, segment_results, mixed_raga_analysis):
        """Plot the segment analysis for mixed raga detection."""
        if not segment_results:
            return
        
        # Create plot for segment analysis
        plt.figure(figsize=(15, 8))
        
        # Track all identified ragas
        all_ragas = set()
        for segment in segment_results:
            if 'top_matches' in segment and segment['top_matches']:
                all_ragas.add(segment['top_matches'][0]['raga_id'])
        
        # Assign colors to ragas
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_ragas)))
        raga_colors = {raga: colors[i] for i, raga in enumerate(all_ragas)}
        
        # Plot segment timeline
        ax = plt.subplot(2, 1, 1)
        
        for i, segment in enumerate(segment_results):
            if 'top_matches' not in segment or not segment['top_matches']:
                continue
                
            start_time = segment['start_time']
            end_time = segment['end_time']
            top_raga = segment['top_matches'][0]['raga_id']
            confidence = segment['top_matches'][0]['confidence']
            
            # Get raga name
            raga_name = segment['top_matches'][0].get('raga_name', top_raga)
            
            # Plot segment bar
            plt.barh(0, end_time - start_time, left=start_time, height=0.5,
                    color=raga_colors[top_raga], alpha=0.7)
            
            # Add raga label
            plt.text((start_time + end_time) / 2, 0, raga_name,
                    ha='center', va='center', fontsize=10)
        
        plt.title("Raga Segments Timeline")
        plt.xlabel("Time (seconds)")
        ax.set_yticks([])
        
        # Plot transition analysis
        plt.subplot(2, 1, 2)
        
        # Extract data for plotting
        ragas = list(all_ragas)
        matrix = np.zeros((len(ragas), len(ragas)))
        
        # Calculate transition counts
        for transition in mixed_raga_analysis.get('transitions', []):
            from_idx = ragas.index(transition['from'])
            to_idx = ragas.index(transition['to'])
            matrix[from_idx, to_idx] += 1
        
        # Plot transition matrix
        plt.imshow(matrix, cmap='Blues')
        plt.colorbar(label='Transition Count')
        plt.title(f"Raga Transitions (Mixed Raga Confidence: {mixed_raga_analysis.get('confidence', 0):.2f})")
        
        # Get raga names for labels
        raga_names = []
        for raga_id in ragas:
            name = raga_id
            for model_id, model in self.feature_extractor.raga_models.items():
                if model_id == raga_id:
                    if 'metadata' in model and 'raga_name' in model['metadata']:
                        name = model['metadata']['raga_name']
                    elif 'name' in model:
                        name = model['name']
                    break
            raga_names.append(name)
        
        plt.xticks(range(len(ragas)), raga_names, rotation=45, ha='right')
        plt.yticks(range(len(ragas)), raga_names)
        
        plt.tight_layout()
        plt.show()
    
    def get_description(self, include_features=True):
        """
        Get a textual description of the raga identification results.
        
        Parameters:
        - include_features: Whether to include specific raga features
        
        Returns:
        - String description of the analysis
        """
        if not self.analysis_results:
            return "No analysis results available."
        
        description = []
        
        # Add title
        if 'file_path' in self.analysis_results:
            description.append(f"Raga Analysis for: {os.path.basename(self.analysis_results['file_path'])}")
            description.append("")
        
        # Check if it's a mixed raga
        is_mixed = False
        if ('mixed_raga_analysis' in self.analysis_results and 
            self.analysis_results['mixed_raga_analysis'].get('is_mixed_raga', False)):
            
            is_mixed = True
            description.append("Analysis indicates this is likely a MIXED RAGA composition.")
            description.append(f"Mixed Raga Confidence: {self.analysis_results['mixed_raga_analysis']['confidence']:.2f}")
            description.append("")
            
            # Add raga transitions
            if 'transitions' in self.analysis_results['mixed_raga_analysis']:
                transitions = self.analysis_results['mixed_raga_analysis']['transitions']
                if transitions:
                    description.append("Detected Raga Transitions:")
                    for transition in transitions:
                        description.append(f"  • {transition['from']} → {transition['to']} (at {transition['segment']})")
                    description.append("")
        
        # Add overall results
        if 'overall_results' in self.analysis_results and 'top_matches' in self.analysis_results['overall_results']:
            if is_mixed:
                description.append("Overall Raga Matches (across entire recording):")
            else:
                description.append("Identified Ragas (in order of confidence):")
                
            for i, match in enumerate(self.analysis_results['overall_results']['top_matches']):
                description.append(f"{i+1}. {match['raga_name']} ({match['raga_id']}) - Confidence: {match['confidence']:.2f}")
                
                # Add similar ragas
                if 'similar_ragas' in match and match['similar_ragas']:
                    description.append("   Similar Ragas:")
                    for similar in match['similar_ragas']:
                        description.append(f"   • {similar['raga_name']} ({similar['raga_id']}) - Similarity: {similar['similarity']:.2f}")
                
                # Add features for top match
                if i == 0 and include_features and 'features' in match:
                    description.append("")
                    description.append(f"Key Features of {match['raga_name']}:")
                    
                    if 'arohana' in match['features']:
                        description.append(f"  • Arohana (Ascending): {match['features']['arohana']}")
                    
                    if 'avarohana' in match['features']:
                        description.append(f"  • Avarohana (Descending): {match['features']['avarohana']}")
                    
                    if 'vadi' in match['features']:
                        description.append(f"  • Vadi (Dominant note): {match['features']['vadi']}")
                    
                    if 'samvadi' in match['features']:
                        description.append(f"  • Samvadi (Second dominant): {match['features']['samvadi']}")
                    
                    if 'time_of_day' in match['features']:
                        description.append(f"  • Traditional time: {match['features']['time_of_day']}")
                
                description.append("")
        
        # Add segment analysis for mixed raga
        if is_mixed and 'segment_results' in self.analysis_results:
            description.append("Segment-by-Segment Analysis:")
            
            for segment in self.analysis_results['segment_results']:
                if 'top_matches' not in segment or not segment['top_matches']:
                    continue
                    
                top_match = segment['top_matches'][0]
                description.append(f"  • {segment['segment_label']} ({segment['start_time']:.1f}s - {segment['end_time']:.1f}s):")
                description.append(f"    Primary Raga: {top_match['raga_name']} ({top_match['raga_id']}) - Confidence: {top_match['confidence']:.2f}")
                
                # Add secondary matches if available
                if len(segment['top_matches']) > 1:
                    description.append("    Secondary Matches:")
                    for match in segment['top_matches'][1:3]:  # Show up to 2 secondary matches
                        description.append(f"      - {match['raga_name']} ({match['raga_id']}) - Confidence: {match['confidence']:.2f}")
                
                description.append("")
        
        # Add disclaimer
        description.append("Note: Raga identification is approximate. Multiple ragas may share similar structures.")
        description.append("      Consider these results as suggestions rather than definitive identifications.")
        
        return "\n".join(description)
    
    def save_results(self, output_path=None):
        """
        Save the analysis results to a JSON file.
        
        Parameters:
        - output_path: Path to save the results, or None for auto-generated
        
        Returns:
        - Path to the saved file
        """
        if not self.analysis_results:
            print("No analysis results to save.")
            return None
        
        # Generate output path if not provided
        if output_path is None:
            if 'file_path' in self.analysis_results:
                base_dir = os.path.dirname(self.analysis_results['file_path'])
                base_name = os.path.splitext(os.path.basename(self.analysis_results['file_path']))[0]
                output_path = os.path.join(base_dir, f"{base_name}_raga_analysis.json")
            else:
                output_path = f"raga_analysis_{int(time.time())}.json"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save the file
        try:
            # Convert NumPy types to Python native types for JSON
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
            
            with open(output_path, 'w') as f:
                json.dump(convert_for_json(self.analysis_results), f, indent=2)
            
            print(f"Saved analysis results to: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error saving results: {e}")
            return None


def identify_raga_in_file(file_path, preprocess=True, segmentation=True, top_n=3, plot=False, save_results=True):
    """
    Utility function to identify the raga(s) in an audio file.
    
    Parameters:
    - file_path: Path to the audio file
    - preprocess: Whether to preprocess the audio
    - segmentation: Whether to analyze segments for mixed ragas
    - top_n: Number of top matches to return
    - plot: Whether to display plots
    - save_results: Whether to save results to a file
    
    Returns:
    - Dictionary with identification results
    """
    # Create identifier
    identifier = RagaIdentifier(segmentation=segmentation)
    
    # Analyze file
    results = identifier.identify_raga(file_path, preprocess=preprocess, top_n=top_n, plot=plot)
    
    if results:
        # Print description
        description = identifier.get_description()
        print("\n" + description)
        
        # Save results if requested
        if save_results:
            identifier.save_results()
    
    return results


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Identify the raga(s) in a music recording.')
    parser.add_argument('file', help='Path to audio file')
    parser.add_argument('--no-preprocess', action='store_false', dest='preprocess', 
                      help='Disable audio preprocessing')
    parser.add_argument('--no-segments', action='store_false', dest='segmentation',
                      help='Disable segmentation for mixed ragas')
    parser.add_argument('-n', '--top-n', type=int, default=3, 
                      help='Number of top matches to return')
    parser.add_argument('-p', '--plot', action='store_true', 
                      help='Display plots during analysis')
    parser.add_argument('--no-save', action='store_false', dest='save_results',
                      help='Do not save results to file')
    
    args = parser.parse_args()
    
    # Identify raga
    identify_raga_in_file(
        args.file,
        preprocess=args.preprocess,
        segmentation=args.segmentation,
        top_n=args.top_n,
        plot=args.plot,
        save_results=args.save_results
    )