#!/usr/bin/env python3
"""
Machine Learning Raga Pattern Generator
--------------------------------------
A system that uses machine learning to generate authentic raga patterns
based on analysis of existing performances.
"""

import os
import numpy as np
import pickle
import json
import random
import mido
from mido import Message, MidiFile, MidiTrack
import time

# For ML model
from sklearn.model_selection import train_test_split
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model, Model
    from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Embedding
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Installing minimal dependencies for basic functionality.")
    TF_AVAILABLE = False

# Fallback to simpler models if TensorFlow not available
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    print("hmmlearn not available. Using simple Markov model.")
    HMM_AVAILABLE = False


class RagaPatternDataset:
    """
    A dataset class for storing and preprocessing raga patterns.
    """
    
    def __init__(self, raga_id=None):
        """
        Initialize the dataset.
        
        Parameters:
        - raga_id: ID of the raga to focus on, or None for all ragas
        """
        self.raga_id = raga_id
        self.sequences = []  # List of note sequences
        self.labels = []  # List of raga IDs for each sequence
        self.raga_names = {}  # Mapping of raga IDs to names
        
        # For preprocessing
        self.max_sequence_length = 32
        self.note_to_int = {}  # Mapping of notes to integers
        self.int_to_note = {}  # Reverse mapping
        self.num_notes = 0  # Number of unique notes
        
        # Processed data
        self.X = None
        self.y = None
        
        # Load raga information if available
        self._load_raga_info()
    
    def _load_raga_info(self):
        """Load basic raga information from ragas.json if available."""
        try:
            with open('data/ragas.json', 'r') as f:
                data = json.load(f)
                # Create mapping of raga IDs to names
                for raga in data['ragas']:
                    self.raga_names[raga['id']] = raga['name']
        except (FileNotFoundError, json.JSONDecodeError):
            print("Raga information file not found or invalid.")
    
    def add_sequence(self, notes, raga_id):
        """
        Add a note sequence to the dataset.
        
        Parameters:
        - notes: List of note values (scale degrees or MIDI notes)
        - raga_id: ID of the raga for this sequence
        """
        if self.raga_id is None or raga_id == self.raga_id:
            self.sequences.append(notes)
            self.labels.append(raga_id)
    
    def add_midi_file(self, filename, raga_id):
        """
        Extract and add note sequences from a MIDI file.
        
        Parameters:
        - filename: Path to the MIDI file
        - raga_id: ID of the raga for this file
        
        Returns:
        - True if successful, False otherwise
        """
        try:
            # Load MIDI file
            midi = MidiFile(filename)
            
            # Extract note sequences from each track
            for track in midi.tracks:
                # Extract notes
                notes = []
                active_notes = {}
                
                for msg in track:
                    if msg.type == 'note_on' and msg.velocity > 0:
                        # Store note-on time
                        active_notes[msg.note] = msg.time
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        # Check if this note is active
                        if msg.note in active_notes:
                            # Add note to sequence
                            notes.append(msg.note)
                            # Remove from active notes
                            del active_notes[msg.note]
                
                # If we extracted a meaningful sequence, add it
                if len(notes) > 8:  # Minimum meaningful length
                    # Convert absolute notes to scale degrees if possible
                    # This helps generalize across octaves and keys
                    scale_degrees = self._convert_to_scale_degrees(notes)
                    if scale_degrees:
                        self.add_sequence(scale_degrees, raga_id)
                    else:
                        self.add_sequence(notes, raga_id)
            
            return True
        except Exception as e:
            print(f"Error processing MIDI file {filename}: {e}")
            return False
    
    def _convert_to_scale_degrees(self, notes):
        """
        Convert absolute MIDI notes to scale degrees.
        
        Parameters:
        - notes: List of MIDI note values
        
        Returns:
        - List of scale degrees, or None if conversion not possible
        """
        # This is a simplified conversion that assumes the first note is the tonic (Sa)
        if not notes:
            return None
        
        # Assume first note is Sa
        tonic = notes[0]
        
        # Convert to scale degrees (semitones above Sa)
        scale_degrees = [(note - tonic) % 12 for note in notes]
        
        return scale_degrees
    
    def preprocess_data(self):
        """
        Preprocess the dataset for ML model training.
        
        Returns:
        - (X, y) tuple of processed data and labels
        """
        if not self.sequences:
            print("No data to preprocess.")
            return None, None
        
        # Create note to integer mapping
        all_notes = set()
        for seq in self.sequences:
            all_notes.update(seq)
        
        self.note_to_int = {note: i for i, note in enumerate(sorted(all_notes))}
        self.int_to_note = {i: note for note, i in self.note_to_int.items()}
        self.num_notes = len(self.note_to_int)
        
        # Process sequences into training data
        # For each sequence, we create multiple training examples
        # where each example is a fixed-length window of notes
        X = []
        y = []
        
        for sequence in self.sequences:
            # Convert notes to integers
            int_sequence = [self.note_to_int[note] for note in sequence]
            
            # Create training examples
            for i in range(len(int_sequence) - self.max_sequence_length):
                # Input is a window of notes
                X.append(int_sequence[i:i + self.max_sequence_length])
                # Target is the next note
                y.append(int_sequence[i + self.max_sequence_length])
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Normalize X values
        X = X / float(self.num_notes)
        
        self.X = X
        self.y = y
        
        return X, y
    
    def save_preprocessed(self, filename='data/raga_dataset.pkl'):
        """
        Save the preprocessed dataset to a file.
        
        Parameters:
        - filename: Output filename
        """
        # Make sure data is preprocessed
        if self.X is None or self.y is None:
            self.preprocess_data()
        
        # Prepare data for saving
        data = {
            'X': self.X,
            'y': self.y,
            'note_to_int': self.note_to_int,
            'int_to_note': self.int_to_note,
            'num_notes': self.num_notes,
            'max_sequence_length': self.max_sequence_length,
            'raga_names': self.raga_names
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save to file
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    
    def load_preprocessed(self, filename='data/raga_dataset.pkl'):
        """
        Load a preprocessed dataset from a file.
        
        Parameters:
        - filename: Input filename
        
        Returns:
        - True if successful, False otherwise
        """
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            self.X = data['X']
            self.y = data['y']
            self.note_to_int = data['note_to_int']
            self.int_to_note = data['int_to_note']
            self.num_notes = data['num_notes']
            self.max_sequence_length = data['max_sequence_length']
            self.raga_names = data.get('raga_names', {})
            
            return True
        except Exception as e:
            print(f"Error loading preprocessed dataset: {e}")
            return False


class LSTMRagaModel:
    """
    LSTM-based model for learning and generating raga patterns.
    """
    
    def __init__(self, dataset):
        """
        Initialize the model.
        
        Parameters:
        - dataset: RagaPatternDataset instance with preprocessed data
        """
        self.dataset = dataset
        self.model = None
        
        # Check if TensorFlow is available
        if not TF_AVAILABLE:
            print("TensorFlow not available. LSTM model cannot be created.")
            self.tf_available = False
            return
        
        self.tf_available = True
    
    def build_model(self):
        """Build the LSTM model architecture."""
        if not self.tf_available:
            print("TensorFlow not available. Cannot build LSTM model.")
            return
        
        # Check if we have preprocessed data
        if self.dataset.X is None or self.dataset.y is None:
            print("No preprocessed data available. Please preprocess the dataset first.")
            return
        
        # Build model architecture
        self.model = Sequential()
        
        # Input shape: (max_sequence_length, 1)
        self.model.add(LSTM(256, input_shape=(self.dataset.max_sequence_length, 1), return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(256))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(self.dataset.num_notes, activation='softmax'))
        
        # Compile model
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
        
        # Print model summary
        self.model.summary()
    
    def train(self, epochs=50, batch_size=64, validation_split=0.2):
        """
        Train the model on the dataset.
        
        Parameters:
        - epochs: Number of training epochs
        - batch_size: Batch size for training
        - validation_split: Fraction of data to use for validation
        
        Returns:
        - Training history
        """
        if not self.tf_available or self.model is None:
            print("LSTM model not available. Cannot train.")
            return None
        
        # Check if we have preprocessed data
        if self.dataset.X is None or self.dataset.y is None:
            print("No preprocessed data available. Please preprocess the dataset first.")
            return None
        
        # Reshape X for LSTM [samples, time steps, features]
        X = np.reshape(self.dataset.X, (self.dataset.X.shape[0], self.dataset.X.shape[1], 1))
        
        # Set up callbacks
        checkpoint = ModelCheckpoint(
            "models/raga_lstm_model.h5",
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min'
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            mode='min'
        )
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Train the model
        history = self.model.fit(
            X, self.dataset.y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[checkpoint, early_stopping]
        )
        
        return history
    
    def load_model(self, filename='models/raga_lstm_model.h5'):
        """
        Load a trained model from a file.
        
        Parameters:
        - filename: Path to the model file
        
        Returns:
        - True if successful, False otherwise
        """
        if not self.tf_available:
            print("TensorFlow not available. Cannot load LSTM model.")
            return False
        
        try:
            self.model = load_model(filename)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def generate_sequence(self, seed_sequence=None, length=64, temperature=1.0):
        """
        Generate a sequence of notes based on the trained model.
        
        Parameters:
        - seed_sequence: Initial sequence to seed the generation, or None for random
        - length: Length of the sequence to generate
        - temperature: Controls randomness (higher = more random)
        
        Returns:
        - Generated sequence of notes
        """
        if not self.tf_available or self.model is None:
            print("LSTM model not available. Cannot generate sequence.")
            return None
        
        # If no seed provided, create a random one
        if seed_sequence is None:
            # Start with random notes from the dataset
            random_start = random.randint(0, len(self.dataset.sequences) - 1)
            random_seq = self.dataset.sequences[random_start]
            
            # Take a segment of the right length
            start_idx = random.randint(0, max(0, len(random_seq) - self.dataset.max_sequence_length))
            seed_sequence = random_seq[start_idx:start_idx + self.dataset.max_sequence_length]
            
            # Ensure we have enough notes
            while len(seed_sequence) < self.dataset.max_sequence_length:
                seed_sequence.append(random.choice(list(self.dataset.note_to_int.keys())))
        
        # Convert seed to integer indices
        seed_ints = [self.dataset.note_to_int.get(note, 0) for note in seed_sequence]
        
        # Ensure seed is the right length
        if len(seed_ints) > self.dataset.max_sequence_length:
            seed_ints = seed_ints[:self.dataset.max_sequence_length]
        elif len(seed_ints) < self.dataset.max_sequence_length:
            # Pad with zeros if too short
            seed_ints = [0] * (self.dataset.max_sequence_length - len(seed_ints)) + seed_ints
        
        # Generate sequence
        generated = []
        pattern = seed_ints
        
        for i in range(length):
            # Prepare input
            x = np.array(pattern) / float(self.dataset.num_notes)
            x = np.reshape(x, (1, len(pattern), 1))
            
            # Predict next note
            prediction = self.model.predict(x, verbose=0)[0]
            
            # Apply temperature to control randomness
            if temperature != 1.0:
                prediction = np.log(prediction) / temperature
                prediction = np.exp(prediction) / np.sum(np.exp(prediction))
            
            # Sample from the prediction
            next_index = self._sample_with_temperature(prediction, temperature)
            
            # Convert index back to note
            next_note = self.dataset.int_to_note[next_index]
            
            # Add to generated sequence
            generated.append(next_note)
            
            # Update pattern for next prediction (remove first element, add new note)
            pattern = pattern[1:] + [next_index]
        
        return generated
    
    def _sample_with_temperature(self, probabilities, temperature):
        """
        Sample an index from a probability array with temperature control.
        
        Parameters:
        - probabilities: Array of probabilities
        - temperature: Controls randomness (higher = more random)
        
        Returns:
        - Sampled index
        """
        if temperature <= 0:
            return np.argmax(probabilities)
        
        # Scale by temperature
        probabilities = np.asarray(probabilities).astype('float64')
        probabilities = np.log(probabilities + 1e-7) / temperature
        exp_probabilities = np.exp(probabilities)
        probabilities = exp_probabilities / np.sum(exp_probabilities)
        
        # Sample
        index = np.random.choice(len(probabilities), p=probabilities)
        
        return index


class MarkovRagaModel:
    """
    Markov chain model for raga pattern generation.
    A simpler alternative when TensorFlow is not available.
    """
    
    def __init__(self, dataset, order=2):
        """
        Initialize the Markov model.
        
        Parameters:
        - dataset: RagaPatternDataset instance
        - order: Order of the Markov chain (how many previous notes to consider)
        """
        self.dataset = dataset
        self.order = order
        self.transitions = {}  # Transition probabilities
    
    def train(self):
        """
        Train the Markov model on the dataset.
        """
        # Check if we have data
        if not self.dataset.sequences:
            print("No data available. Cannot train Markov model.")
            return
        
        # Build transition probabilities
        for sequence in self.dataset.sequences:
            # Use sequence directly (no need to convert to integers)
            for i in range(len(sequence) - self.order):
                # Current state is a tuple of 'order' consecutive notes
                current_state = tuple(sequence[i:i + self.order])
                # Next note
                next_note = sequence[i + self.order]
                
                # Update transitions
                if current_state not in self.transitions:
                    self.transitions[current_state] = {}
                
                if next_note not in self.transitions[current_state]:
                    self.transitions[current_state][next_note] = 0
                
                self.transitions[current_state][next_note] += 1
        
        # Convert counts to probabilities
        for state, nexts in self.transitions.items():
            total = sum(nexts.values())
            for note in nexts:
                nexts[note] = nexts[note] / total
    
    def generate_sequence(self, seed_sequence=None, length=64):
        """
        Generate a sequence of notes based on the Markov model.
        
        Parameters:
        - seed_sequence: Initial sequence to seed the generation, or None for random
        - length: Length of the sequence to generate
        
        Returns:
        - Generated sequence of notes
        """
        if not self.transitions:
            print("Markov model not trained. Cannot generate sequence.")
            return None
        
        # If no seed provided, create a random one
        if seed_sequence is None:
            # Start with a random state from the transitions
            if self.transitions:
                current_state = random.choice(list(self.transitions.keys()))
            else:
                # If no transitions, create a random state
                notes = list(set(note for seq in self.dataset.sequences for note in seq))
                if not notes:
                    return None
                current_state = tuple(random.choices(notes, k=self.order))
        else:
            # Use provided seed sequence
            if len(seed_sequence) < self.order:
                # Pad seed if necessary
                notes = list(set(note for seq in self.dataset.sequences for note in seq))
                padding = random.choices(notes, k=self.order - len(seed_sequence))
                seed_sequence = padding + seed_sequence
            
            current_state = tuple(seed_sequence[-self.order:])
        
        # Generate sequence
        generated = list(current_state)
        
        for _ in range(length):
            # Check if current state is in transitions
            if current_state in self.transitions:
                # Get possible next notes and their probabilities
                next_notes = list(self.transitions[current_state].keys())
                probabilities = list(self.transitions[current_state].values())
                
                # Sample next note
                next_note = random.choices(next_notes, weights=probabilities, k=1)[0]
            else:
                # If state not found, choose a random note from the dataset
                notes = list(set(note for seq in self.dataset.sequences for note in seq))
                next_note = random.choice(notes)
            
            # Add to generated sequence
            generated.append(next_note)
            
            # Update current state
            current_state = tuple(generated[-self.order:])
        
        # Return only the newly generated notes (not the seed)
        return generated[self.order:]
    
    def save_model(self, filename='models/raga_markov_model.pkl'):
        """
        Save the trained model to a file.
        
        Parameters:
        - filename: Output filename
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save the transitions dictionary
        with open(filename, 'wb') as f:
            pickle.dump({
                'order': self.order,
                'transitions': self.transitions
            }, f)
    
    def load_model(self, filename='models/raga_markov_model.pkl'):
        """
        Load a trained model from a file.
        
        Parameters:
        - filename: Path to the model file
        
        Returns:
        - True if successful, False otherwise
        """
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            self.order = data['order']
            self.transitions = data['transitions']
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


class MLRagaGenerator:
    """
    Main class for generating raga patterns using machine learning models.
    """
    
    def __init__(self, model_type='markov'):
        """
        Initialize the generator.
        
        Parameters:
        - model_type: Type of model to use ('lstm' or 'markov')
        """
        self.model_type = model_type
        self.dataset = RagaPatternDataset()
        self.model = None
        
        # Check if LSTM is available
        if model_type == 'lstm' and not TF_AVAILABLE:
            print("TensorFlow not available. Falling back to Markov model.")
            self.model_type = 'markov'
    
    def load_data(self, midi_dir='data/midi_corpus', raga_mappings=None):
        """
        Load data from MIDI files in a directory.
        
        Parameters:
        - midi_dir: Directory containing MIDI files
        - raga_mappings: Dictionary mapping filenames to raga IDs
        
        Returns:
        - Number of files successfully loaded
        """
        if not os.path.exists(midi_dir):
            print(f"Directory {midi_dir} does not exist.")
            return 0
        
        # Default mappings if none provided
        if raga_mappings is None:
            # Try to infer raga from filename
            raga_mappings = {}
            
            # Try to load mappings from file if available
            try:
                with open(os.path.join(midi_dir, 'raga_mappings.json'), 'r') as f:
                    raga_mappings = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                print("No raga mappings found. Will try to infer from filenames.")
        
        # Load MIDI files
        files_loaded = 0
        for filename in os.listdir(midi_dir):
            if filename.endswith('.mid') or filename.endswith('.midi'):
                path = os.path.join(midi_dir, filename)
                
                # Determine the raga for this file
                raga_id = raga_mappings.get(filename)
                if raga_id is None:
                    # Try to infer from filename
                    for raga_id in self.dataset.raga_names:
                        if raga_id in filename.lower():
                            break
                    else:
                        # If no match, use a generic ID
                        raga_id = 'unknown'
                
                # Add the file to the dataset
                if self.dataset.add_midi_file(path, raga_id):
                    files_loaded += 1
        
        print(f"Loaded {files_loaded} MIDI files.")
        return files_loaded
    
    def prepare_model(self, force_retrain=False):
        """
        Prepare the ML model (load or train).
        
        Parameters:
        - force_retrain: Whether to force retraining even if a model exists
        
        Returns:
        - True if model is ready, False otherwise
        """
        # Check if we have data
        if not self.dataset.sequences:
            print("No data available. Cannot prepare model.")
            return False
        
        # Preprocess data if needed
        if self.dataset.X is None or self.dataset.y is None:
            self.dataset.preprocess_data()
        
        # Create or load the model
        if self.model_type == 'lstm' and TF_AVAILABLE:
            self.model = LSTMRagaModel(self.dataset)
            model_file = 'models/raga_lstm_model.h5'
        else:
            self.model = MarkovRagaModel(self.dataset)
            model_file = 'models/raga_markov_model.pkl'
        
        # Try to load existing model if not forcing retrain
        if not force_retrain and os.path.exists(model_file):
            print(f"Loading existing model from {model_file}")
            if self.model.load_model(model_file):
                return True
            else:
                print("Failed to load model. Will train a new one.")
        
        # Train new model
        print("Training new model...")
        if self.model_type == 'lstm' and TF_AVAILABLE:
            self.model.build_model()
            self.model.train(epochs=30, batch_size=64)
        else:
            self.model.train()
            self.model.save_model()
        
        return True
    
    def generate_melody(self, seed=None, length=64, raga_id=None, save_midi=True):
        """
        Generate a melody based on the trained model.
        
        Parameters:
        - seed: Initial sequence to seed the generation, or None for random
        - length: Length of the sequence to generate
        - raga_id: Raga ID to use for generation context
        - save_midi: Whether to save the generated melody as a MIDI file
        
        Returns:
        - Generated sequence of notes, and MIDI filename if saved
        """
        if self.model is None:
            print("Model not prepared. Call prepare_model() first.")
            return None, None
        
        # If raga_id specified but no seed, try to find a seed from that raga
        if seed is None and raga_id is not None:
            # Find sequences for this raga
            raga_sequences = []
            for i, label in enumerate(self.dataset.labels):
                if label == raga_id:
                    raga_sequences.append(self.dataset.sequences[i])
            
            # If we have sequences for this raga, use one as seed
            if raga_sequences:
                seed_seq = random.choice(raga_sequences)
                start_idx = random.randint(0, max(0, len(seed_seq) - 8))
                seed = seed_seq[start_idx:start_idx + 8]
        
        # Generate sequence
        if self.model_type == 'lstm' and isinstance(self.model, LSTMRagaModel):
            generated = self.model.generate_sequence(seed, length, temperature=1.0)
        else:
            generated = self.model.generate_sequence(seed, length)
        
        # Save as MIDI if requested
        midi_filename = None
        if save_midi and generated:
            midi_filename = self._create_midi(generated, raga_id)
        
        return generated, midi_filename
    
    def _create_midi(self, notes, raga_id=None):
        """
        Create a MIDI file from a sequence of notes.
        
        Parameters:
        - notes: Sequence of notes (scale degrees or MIDI notes)
        - raga_id: Raga ID for filename
        
        Returns:
        - Filename of the generated MIDI file
        """
        # Create MIDI file
        midi = MidiFile()
        track = MidiTrack()
        midi.tracks.append(track)
        
        # Add tempo (moderate for lo-fi)
        tempo = mido.bpm2tempo(75)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))
        
        # Add track name
        raga_name = self.dataset.raga_names.get(raga_id, "Unknown")
        track.append(mido.MetaMessage('track_name', name=f"ML Generated {raga_name} Melody"))
        
        # Add time signature (4/4 for lo-fi)
        track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))
        
        # Determine if notes are scale degrees or absolute
        is_scale_degrees = all(isinstance(note, int) and note < 12 for note in notes)
        
        # Convert to absolute MIDI notes if needed
        if is_scale_degrees:
            # Use C4 (MIDI note 60) as Sa
            base_note = 60
            midi_notes = [base_note + note for note in notes]
        else:
            # Use notes as is
            midi_notes = notes
        
        # Add notes
        ticks_per_beat = midi.ticks_per_beat
        duration = ticks_per_beat // 2  # Eighth notes
        
        for note in midi_notes:
            # Add some velocity variation
            velocity = random.randint(65, 95)
            
            # Note on
            track.append(Message('note_on', note=note, velocity=velocity, time=0))
            
            # Note off
            track.append(Message('note_off', note=note, velocity=0, time=duration))
        
        # Create filename
        timestamp = int(time.time())
        if raga_id:
            filename = f"outputs/ml_generated_{raga_id}_{timestamp}.mid"
        else:
            filename = f"outputs/ml_generated_melody_{timestamp}.mid"
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save file
        midi.save(filename)
        
        return filename

    def generate_complete_track(self, raga_id=None, length=32, base_note=60, bpm=75):
        """
        Generate a complete track with melody, chords, and rhythm.
        
        Parameters:
        - raga_id: Raga ID to use for generation context
        - length: Length of the melody to generate
        - base_note: Base MIDI note for Sa
        - bpm: Tempo in beats per minute
        
        Returns:
        - Dictionary with filenames for each component
        """
        # Make sure model is prepared
        if self.model is None and not self.prepare_model():
            print("Failed to prepare model.")
            return None
        
        # Generate melody
        melody_notes, melody_file = self.generate_melody(
            length=length,
            raga_id=raga_id,
            save_midi=True
        )
        
        # For now, use traditional methods for chords and rhythm
        # In the future, we could train separate ML models for these
        
        # Create a basic dictionary of results
        result = {
            'melody': melody_file,
            'raga': self.dataset.raga_names.get(raga_id, "Unknown"),
            'bpm': bpm
        }
        
        return result


def initialize_ml_system():
    """
    Initialize the ML-based raga generator system.
    
    Returns:
    - Prepared MLRagaGenerator instance, or None if initialization failed
    """
    # Check if TensorFlow is available
    model_type = 'lstm' if TF_AVAILABLE else 'markov'
    
    # Create generator
    generator = MLRagaGenerator(model_type)
    
    # Check if we have a preprocessed dataset
    if os.path.exists('data/raga_dataset.pkl'):
        print("Loading preprocessed dataset...")
        if generator.dataset.load_preprocessed():
            print("Dataset loaded successfully.")
        else:
            print("Failed to load dataset.")
            return None
    else:
        print("No preprocessed dataset found.")
        print("Please load MIDI data using generator.load_data()")
        return generator
    
    # Prepare model
    if generator.prepare_model():
        print("Model prepared successfully.")
        return generator
    else:
        print("Failed to prepare model.")
        return None

# Example usage
if __name__ == "__main__":
    # Initialize the system
    generator = initialize_ml_system()
    
    if generator is None:
        print("Failed to initialize ML system.")
    else:
        # Generate a melody
        print("Generating a melody...")
        notes, midi_file = generator.generate_melody(length=64)
        
        if midi_file:
            print(f"Generated melody saved to {midi_file}")
        else:
            print("Failed to generate melody.")