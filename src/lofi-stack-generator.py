#!/usr/bin/env python3
"""
LofiStackGenerator
------------------
A system for generating lo-fi beats based on Indian ragas, 
inspired by Splice's Stack but specialized for raga-based music.
"""

import os
import json
import random
import time
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.effects import low_pass_filter, high_pass_filter

# Import from existing modules
from enhanced_raga_generator import EnhancedRagaGenerator
from melodic_pattern_generator import MelodicPatternGenerator
from ambient_texture_manager import AmbientTextureManager

class LofiStackGenerator:
    """Generate complete lo-fi beats based on ragas or input samples."""
    
    def __init__(self, sample_library_path="sample_library", output_dir="outputs"):
        """
        Initialize the generator.
        
        Parameters:
        - sample_library_path: Path to the lo-fi sample library
        - output_dir: Directory for output files
        """
        self.sample_library_path = sample_library_path
        self.output_dir = output_dir
        
        # Ensure directories exist
        os.makedirs(sample_library_path, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize generators
        self.raga_generator = EnhancedRagaGenerator()
        self.melodic_generator = MelodicPatternGenerator()
        
        # Initialize ambient texture manager
        self.texture_manager = AmbientTextureManager(os.path.join(sample_library_path, "textures"))
        
        # Load sample library structure
        self._load_sample_library()
        
        # Track settings
        self.bpm = 75  # Default lo-fi tempo
        self.key = "C"  # Default key
        self.current_raga = None
        self.session_id = int(time.time())
        
        print(f"LofiStackGenerator initialized with session ID: {self.session_id}")
    
    def _load_sample_library(self):
        """Load and catalog the sample library."""
        self.samples = {
            "drums": {
                "kicks": [],
                "snares": [],
                "hats": [],
                "percs": [],
                "loops": []
            },
            "effects": {
                "vinyl": [],
                "foley": [],
                "ambience": []
            },
            "oneshots": {
                "keys": [],
                "guitar": [],
                "bass": []
            }
        }
        
        # Check if the library exists
        if not os.path.exists(self.sample_library_path):
            print(f"Sample library not found at {self.sample_library_path}. Creating empty structure.")
            self._create_sample_library_structure()
            return
        
        # Scan directory for samples
        for category in self.samples:
            category_path = os.path.join(self.sample_library_path, category)
            if os.path.exists(category_path):
                for subcategory in self.samples[category]:
                    subcategory_path = os.path.join(category_path, subcategory)
                    if os.path.exists(subcategory_path):
                        # Find audio files
                        samples = [
                            os.path.join(subcategory_path, f) 
                            for f in os.listdir(subcategory_path)
                            if f.endswith(('.wav', '.mp3', '.aiff', '.ogg'))
                        ]
                        self.samples[category][subcategory] = samples
        
        # Count total samples
        total_samples = sum(len(files) for category in self.samples.values() 
                           for files in category.values())
        
        print(f"Loaded {total_samples} samples from library")
    
    def _create_sample_library_structure(self):
        """Create the initial sample library directory structure."""
        for category in self.samples:
            for subcategory in self.samples[category]:
                path = os.path.join(self.sample_library_path, category, subcategory)
                os.makedirs(path, exist_ok=True)
        
        # Create a readme file
        readme_path = os.path.join(self.sample_library_path, "README.md")
        with open(readme_path, 'w') as f:
            f.write("# Lo-Fi Sample Library\n\n")
            f.write("Add your lo-fi samples to the respective folders:\n\n")
            for category in self.samples:
                f.write(f"## {category.capitalize()}\n\n")
                for subcategory in self.samples[category]:
                    f.write(f"- {subcategory}\n")
                f.write("\n")
    
    def set_bpm(self, bpm):
        """Set the beats per minute for generation."""
        self.bpm = bpm
        print(f"BPM set to {bpm}")
        return self
    
    def set_key(self, key):
        """Set the musical key for generation."""
        self.key = key
        print(f"Key set to {key}")
        return self
    
    def set_raga(self, raga_id):
        """Set the raga for melody generation."""
        if raga_id in self.raga_generator.ragas_data:
            self.current_raga = raga_id
            raga_name = self.raga_generator.ragas_data[raga_id]['name']
            print(f"Raga set to {raga_name} ({raga_id})")
            return True
        else:
            print(f"Raga '{raga_id}' not found in database")
            return False
    
    def generate_from_raga(self, raga_id=None, length=32, components=None):
        """
        Generate a complete lo-fi beat based on a raga.
        
        Parameters:
        - raga_id: ID of the raga to use (if None, uses current_raga)
        - length: Length of the melody in notes
        - components: Dictionary specifying which components to generate
                    (defaults to all: melody, chords, bass, drums)
        
        Returns:
        - Dictionary with paths to generated files
        """
        # Set raga if provided
        if raga_id:
            if not self.set_raga(raga_id):
                return None
        elif not self.current_raga:
            print("No raga selected. Please call set_raga() first or provide raga_id.")
            return None
        
        # Default components
        if components is None:
            components = {
                "melody": True,
                "chords": True,
                "bass": True,
                "drums": True,
                "effects": True
            }
        
        # Create output directory for this generation
        timestamp = int(time.time())
        output_name = f"{self.current_raga}_{timestamp}"
        beat_dir = os.path.join(self.output_dir, output_name)
        os.makedirs(beat_dir, exist_ok=True)
        
        # Dictionary to store generated file paths
        generated_files = {
            "raga_id": self.current_raga,
            "raga_name": self.raga_generator.ragas_data[self.current_raga]['name'],
            "bpm": self.bpm,
            "key": self.key,
            "timestamp": timestamp,
            "files": {}
        }
        
        # Calculate MIDI base note from key
        base_note = self._key_to_midi_note(self.key)
        
        # Step 1: Generate MIDI components
        midi_files = {}
        
        # Generate melody
        if components.get("melody", True):
            midi_files["melody"] = self.raga_generator.generate_melody(
                self.current_raga,
                length=length,
                use_patterns=True,
                base_note=base_note,
                bpm=self.bpm
            )
            generated_files["files"]["melody_midi"] = midi_files["melody"]
        
        # Generate chord progression
        if components.get("chords", True):
            midi_files["chords"] = self.raga_generator.generate_chord_progression(
                self.current_raga,
                length=4,  # 4-bar chord progression
                base_note=base_note - 12,  # One octave lower
                bpm=self.bpm
            )
            generated_files["files"]["chords_midi"] = midi_files["chords"]
        
        # Generate bass line
        if components.get("bass", True):
            midi_files["bass"] = self.raga_generator.generate_bass_line(
                self.current_raga,
                length=length,
                base_note=base_note - 24,  # Two octaves lower
                bpm=self.bpm
            )
            generated_files["files"]["bass_midi"] = midi_files["bass"]
        
        # Step 2: Generate audio components
        
        # Step 2.1: Select drum samples
        if components.get("drums", True):
            drum_files = self._select_drum_samples()
            generated_files["files"]["drums"] = drum_files
        
        # Step 2.2: Select effects
        if components.get("effects", True):
            effect_files = self._select_effect_samples()
            generated_files["files"]["effects"] = effect_files
            # Add to the result dictionary
        if "effects" in generated_files["files"] and "textures" in generated_files["files"]["effects"]:
            textures = generated_files["files"]["effects"]["textures"]
            if textures:
                # Get mood categories for the textures
                texture_moods = []
                for texture_path in textures:
                    for mood, paths in self.texture_manager.textures.items():
                        if any(os.path.samefile(texture_path, p) for p in paths if os.path.exists(p)):
                            texture_moods.append(mood)
                            break
                
                generated_files["texture_moods"] = texture_moods
        
        # Step 3: Create project files for DAW import
        project_info = self._create_project_files(beat_dir, generated_files)
        generated_files["files"]["project"] = project_info
        
        # Save generation info
        info_file = os.path.join(beat_dir, "generation_info.json")
        with open(info_file, 'w') as f:
            json.dump(generated_files, f, indent=2)
        
        print(f"Generated lo-fi beat in {beat_dir}")
        print(f"Info saved to {info_file}")
        
        return generated_files
    
    def generate_from_melody(self, melody_file, identify_raga=True, components=None):
        """
        Generate a lo-fi beat based on an input melody file.
        
        Parameters:
        - melody_file: Path to MIDI or audio melody file
        - identify_raga: Whether to attempt raga identification
        - components: Dictionary specifying which components to generate
        
        Returns:
        - Dictionary with paths to generated files
        """
        # Ensure file exists
        if not os.path.exists(melody_file):
            print(f"Error: Melody file not found: {melody_file}")
            return None
        
        # Default components (exclude melody since it's provided)
        if components is None:
            components = {
                "melody": False,  # Don't generate melody
                "chords": True,
                "bass": True,
                "drums": True,
                "effects": True
            }
        
        # Create output directory
        timestamp = int(time.time())
        output_name = f"custom_melody_{timestamp}"
        beat_dir = os.path.join(self.output_dir, output_name)
        os.makedirs(beat_dir, exist_ok=True)
        
        # Dictionary to store generated file paths
        generated_files = {
            "input_melody": melody_file,
            "bpm": self.bpm,
            "key": self.key,
            "timestamp": timestamp,
            "files": {}
        }
        
        # Step 1: Analyze the melody if requested
        identified_raga = None
        if identify_raga:
            identified_raga = self._identify_raga_from_melody(melody_file)
            if identified_raga:
                self.set_raga(identified_raga['raga_id'])
                generated_files["raga_id"] = identified_raga['raga_id']
                generated_files["raga_name"] = identified_raga['raga_name']
                generated_files["raga_confidence"] = identified_raga['confidence']
        
        # Copy the input melody
        if melody_file.endswith(('.mid', '.midi')):
            # It's already a MIDI file
            melody_copy = os.path.join(beat_dir, "input_melody.mid")
            import shutil
            shutil.copy2(melody_file, melody_copy)
            generated_files["files"]["melody_midi"] = melody_copy
        else:
            # Assume it's an audio file, we might need to convert it
            # For now, just copy it
            melody_copy = os.path.join(beat_dir, f"input_melody{os.path.splitext(melody_file)[1]}")
            import shutil
            shutil.copy2(melody_file, melody_copy)
            generated_files["files"]["melody_audio"] = melody_copy
        
        # Calculate MIDI base note from key
        base_note = self._key_to_midi_note(self.key)
        
        # Step 2: Generate additional MIDI components
        midi_files = {}
        
        # If we have an identified raga, use it; otherwise, use generic generation
        use_raga = self.current_raga if hasattr(self, 'current_raga') and self.current_raga else None
        
        # Generate chord progression
        if components.get("chords", True):
            if use_raga:
                midi_files["chords"] = self.raga_generator.generate_chord_progression(
                    use_raga,
                    length=4,
                    base_note=base_note - 12,
                    bpm=self.bpm
                )
            else:
                # Generate generic chord progression based on key
                midi_files["chords"] = self._generate_generic_chord_progression(
                    self.key, 
                    base_note - 12, 
                    self.bpm
                )
            
            generated_files["files"]["chords_midi"] = midi_files["chords"]
        
        # Generate bass line
        if components.get("bass", True):
            if use_raga:
                midi_files["bass"] = self.raga_generator.generate_bass_line(
                    use_raga,
                    length=32,
                    base_note=base_note - 24,
                    bpm=self.bpm
                )
            else:
                # Generate generic bass line based on key
                midi_files["bass"] = self._generate_generic_bass_line(
                    self.key,
                    base_note - 24,
                    self.bpm
                )
            
            generated_files["files"]["bass_midi"] = midi_files["bass"]
        
        # Step 3: Generate audio components
        
        # Step 3.1: Select drum samples
        if components.get("drums", True):
            drum_files = self._select_drum_samples()
            generated_files["files"]["drums"] = drum_files
        
        # Step 3.2: Select effects
        if components.get("effects", True):
            effect_files = self._select_effect_samples()
            generated_files["files"]["effects"] = effect_files
        
        # Step 4: Create project files for DAW import
        project_info = self._create_project_files(beat_dir, generated_files)
        generated_files["files"]["project"] = project_info
        
        # Save generation info
        info_file = os.path.join(beat_dir, "generation_info.json")
        with open(info_file, 'w') as f:
            json.dump(generated_files, f, indent=2)
        
        print(f"Generated lo-fi beat based on input melody in {beat_dir}")
        print(f"Info saved to {info_file}")
        
        return generated_files
    
    def generate_from_loop(self, loop_file, components=None):
        """
        Generate a lo-fi beat based on an input loop file.
        
        Parameters:
        - loop_file: Path to audio loop file
        - components: Dictionary specifying which components to generate
        
        Returns:
        - Dictionary with paths to generated files
        """
        # This essentially wraps generate_from_melody but with loop-specific handling
        return self.generate_from_melody(loop_file, identify_raga=True, components=components)
    
    def export_to_stems(self, generation_info, format="wav"):
        """
        Export the generated beat to individual stem files.
        
        Parameters:
        - generation_info: Dictionary with generation information
        - format: Audio format for export ('wav', 'mp3', 'ogg')
        
        Returns:
        - Dictionary with paths to stem files
        """
        # TODO: Implement stem export logic
        pass
    
    def render_full_track(self, generation_info, format="wav"):
        """
        Render the complete track by mixing all components.
        
        Parameters:
        - generation_info: Dictionary with generation information
        - format: Audio format for export ('wav', 'mp3', 'ogg')
        
        Returns:
        - Path to the rendered track
        """
        # TODO: Implement full track rendering
        pass
    
    def _select_drum_samples(self):
        """
        Select appropriate drum samples for lo-fi beat.
        
        Returns:
        - Dictionary with selected drum sample paths
        """
        selected_drums = {}
        
        # Select a kick
        if self.samples["drums"]["kicks"]:
            selected_drums["kick"] = random.choice(self.samples["drums"]["kicks"])
        
        # Select a snare
        if self.samples["drums"]["snares"]:
            selected_drums["snare"] = random.choice(self.samples["drums"]["snares"])
        
        # Select hi-hats
        if self.samples["drums"]["hats"]:
            selected_drums["closed_hat"] = random.choice(self.samples["drums"]["hats"])
            selected_drums["open_hat"] = random.choice(self.samples["drums"]["hats"])
        
        # Select percussion
        if self.samples["drums"]["percs"]:
            selected_drums["perc"] = random.choice(self.samples["drums"]["percs"])
        
        # Alternatively, select a full drum loop
        if self.samples["drums"]["loops"] and random.random() < 0.3:  # 30% chance to use a loop
            selected_drums["loop"] = random.choice(self.samples["drums"]["loops"])
        
        return selected_drums
    
    # Then update the _select_effect_samples method to include ambient textures:
    def _select_effect_samples(self):
        """
        Select appropriate effect samples and ambient textures for lo-fi aesthetics.
        
        Returns:
        - Dictionary with selected effect sample paths
        """
        selected_effects = {}
        
        # Select vinyl crackle
        if self.samples["effects"]["vinyl"]:
            selected_effects["vinyl"] = random.choice(self.samples["effects"]["vinyl"])
        
        # Select ambient sound
        if self.samples["effects"]["ambience"] and random.random() < 0.6:  # 60% chance
            selected_effects["ambience"] = random.choice(self.samples["effects"]["ambience"])
        
        # Select foley sound
        if self.samples["effects"]["foley"] and random.random() < 0.4:  # 40% chance
            selected_effects["foley"] = random.choice(self.samples["effects"]["foley"])
        
        # Select ambient textures based on raga
        if hasattr(self, 'current_raga') and self.current_raga:
            textures = self.texture_manager.select_textures_for_raga(self.current_raga, count=2)
            
            if textures:
                # Process the textures for the current project
                processed_textures = []
                
                for i, texture_path in enumerate(textures):
                    # Generate output path in the project directory
                    timestamp = int(time.time())
                    texture_name = os.path.splitext(os.path.basename(texture_path))[0]
                    output_path = f"outputs/{self.current_raga}_{texture_name}_{timestamp}.wav"
                    
                    # Process the texture
                    processed_path = self.texture_manager.process_texture(
                        texture_path,
                        target_path=output_path,
                        duration=60,  # 1 minute
                        volume_db=-18  # Quiet enough to sit in the background
                    )
                    
                    if processed_path:
                        processed_textures.append(processed_path)
                
                if processed_textures:
                    selected_effects["textures"] = processed_textures
        
        return selected_effects
    
    def _create_project_files(self, beat_dir, generation_info):
        """
        Create project files for easy import into DAWs.
        
        Parameters:
        - beat_dir: Directory to save project files
        - generation_info: Dictionary with generation information
        
        Returns:
        - Dictionary with paths to project files
        """
        # For now, create a simple text file with instructions
        project_info = {}
        
        # Create a markdown file with instructions
        md_file = os.path.join(beat_dir, "import_instructions.md")
        with open(md_file, 'w') as f:
            f.write(f"# Lo-Fi Beat Project: {generation_info.get('raga_name', 'Custom')}\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Settings\n\n")
            f.write(f"- BPM: {generation_info['bpm']}\n")
            f.write(f"- Key: {generation_info['key']}\n")
            if 'raga_id' in generation_info:
                f.write(f"- Raga: {generation_info['raga_name']} ({generation_info['raga_id']})\n")
            f.write("\n## Files\n\n")
            
            for category, files in generation_info['files'].items():
                if isinstance(files, dict):
                    f.write(f"### {category.capitalize()}\n\n")
                    for name, path in files.items():
                        f.write(f"- {name}: `{os.path.basename(path)}`\n")
                elif isinstance(files, str):
                    f.write(f"- {category}: `{os.path.basename(files)}`\n")
            
            f.write("\n## Recommended Processing\n\n")
            f.write("### Melody\n")
            f.write("- Light reverb (30-40% wet)\n")
            f.write("- High cut filter around 8-10kHz\n")
            f.write("- Mild chorus effect\n\n")
            
            f.write("### Chords\n")
            f.write("- RC-20 plugin for vintage character\n")
            f.write("- Low cut at 200Hz\n")
            f.write("- Side-chain compression against kick\n\n")
            
            f.write("### Bass\n")
            f.write("- Compression with 4:1 ratio\n")
            f.write("- Slight saturation\n\n")
            
            f.write("### Drums\n")
            f.write("- MPC-style swing (around 55-60%)\n")
            f.write("- Light compression on drum bus\n")
            f.write("- Consider layering a vinyl drum loop\n")
        
        project_info["instructions"] = md_file
        
        # TODO: Add more DAW-specific project files
        
        return project_info
    
    def _identify_raga_from_melody(self, melody_file):
        """
        Attempt to identify the raga from an input melody.
        
        Parameters:
        - melody_file: Path to MIDI or audio melody file
        
        Returns:
        - Dictionary with identified raga information, or None if unsuccessful
        """
        # TODO: Implement more sophisticated raga identification
        # For now, just provide a placeholder implementation
        
        # Check if file exists
        if not os.path.exists(melody_file):
            return None
        
        # If it's a MIDI file, extract notes
        if melody_file.endswith(('.mid', '.midi')):
            try:
                import mido
                midi_data = mido.MidiFile(melody_file)
                
                # Extract all notes from all tracks
                all_notes = []
                for track in midi_data.tracks:
                    for msg in track:
                        if msg.type == 'note_on' and msg.velocity > 0:
                            all_notes.append(msg.note % 12)  # Keep only the pitch class
                
                # Count occurrences of each note
                note_counts = {}
                for note in all_notes:
                    if note in note_counts:
                        note_counts[note] += 1
                    else:
                        note_counts[note] = 1
                
                # Calculate note set
                note_set = sorted(note_counts.keys())
                
                # Find ragas that match this note set
                matches = []
                for raga_id, raga in self.raga_generator.ragas_data.items():
                    if 'arohan' in raga and 'avarohan' in raga:
                        raga_notes = sorted(set([n % 12 for n in raga['arohan'] + raga['avarohan']]))
                        
                        # Calculate how well the note sets match
                        common_notes = set(note_set).intersection(set(raga_notes))
                        
                        # Calculate both coverage metrics
                        # 1. How much of the raga is covered by the melody
                        raga_coverage = len(common_notes) / len(raga_notes) if raga_notes else 0
                        
                        # 2. How much of the melody fits within the raga
                        melody_coverage = len(common_notes) / len(note_set) if note_set else 0
                        
                        # Combined score (weighted average)
                        score = 0.7 * melody_coverage + 0.3 * raga_coverage
                        
                        matches.append((raga_id, score))
                
                # Sort by score (descending)
                matches.sort(key=lambda x: x[1], reverse=True)
                
                # Return top match if score is reasonable
                if matches and matches[0][1] > 0.5:
                    top_raga_id = matches[0][0]
                    return {
                        'raga_id': top_raga_id,
                        'raga_name': self.raga_generator.ragas_data[top_raga_id]['name'],
                        'confidence': matches[0][1],
                        'note_set': note_set
                    }
                
            except Exception as e:
                print(f"Error analyzing MIDI file: {e}")
        
        # If analysis failed or it's an audio file, try a random known raga
        # This is a fallback until more sophisticated audio analysis is implemented
        fallback_ragas = ['yaman', 'bhairav', 'darbari', 'kedar']
        available_ragas = [r for r in fallback_ragas if r in self.raga_generator.ragas_data]
        
        if available_ragas:
            random_raga = random.choice(available_ragas)
            return {
                'raga_id': random_raga,
                'raga_name': self.raga_generator.ragas_data[random_raga]['name'],
                'confidence': 0.5,
                'note_set': []
            }
        
        return None
    
    def _generate_generic_chord_progression(self, key, base_note, bpm):
        """
        Generate a generic chord progression based on the key.
        
        Parameters:
        - key: Musical key (e.g., 'C', 'F#')
        - base_note: MIDI base note
        - bpm: Tempo in beats per minute
        
        Returns:
        - Path to generated MIDI file
        """
        import mido
        from mido import Message, MidiFile, MidiTrack
        
        # Create MIDI file
        midi = MidiFile()
        track = MidiTrack()
        midi.tracks.append(track)
        
        # Add tempo
        tempo = mido.bpm2tempo(bpm)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))
        
        # Add track name
        track.append(mido.MetaMessage('track_name', name=f"Lo-Fi Chords in {key}"))
        
        # Define common lo-fi chord progressions relative to the key
        progressions = [
            [0, 3, 4, 3],  # I-IV-V-IV (very common in lo-fi)
            [0, 5, 3, 4],  # I-vi-IV-V
            [5, 3, 0, 4],  # vi-IV-I-V
            [0, 4, 5, 3],  # I-V-vi-IV
            [0, 3, 5, 4],  # I-IV-vi-V
        ]
        
        # Choose a progression
        progression = random.choice(progressions)
        
        # Define the diatonic chord quality (major/minor)
        qualities = [
            [0, 4, 7],  # Major (I)
            [0, 3, 7],  # Minor (ii)
            [0, 3, 7],  # Minor (iii)
            [0, 4, 7],  # Major (IV)
            [0, 4, 7],  # Major (V)
            [0, 3, 7],  # Minor (vi)
            [0, 3, 6],  # Diminished (viio)
        ]
        
        # Major scale steps
        major_scale = [0, 2, 4, 5, 7, 9, 11]
        
        # Calculate diatonic notes based on key
        key_offset = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }
        
        root_offset = key_offset.get(key, 0)
        
        # Duration for one bar (4 beats)
        ticks_per_beat = midi.ticks_per_beat
        bar_duration = ticks_per_beat * 4
        
        # Generate 4-bar progression, repeated twice
        for _ in range(2):
            for chord_idx in progression:
                # Get chord root and quality
                root = (major_scale[chord_idx] + root_offset) % 12
                chord_quality = qualities[chord_idx]
                
                # Build chord
                chord_notes = []
                for note_offset in chord_quality:
                    note = base_note + root + note_offset
                    chord_notes.append(note)
                
                # Add lo-fi character: sometimes remove a note from the chord
                if random.random() < 0.3:  # 30% chance
                    if len(chord_notes) > 2:  # Ensure we have at least a dyad
                        chord_notes.pop(random.randint(0, len(chord_notes) - 1))
                
                # Note on events (all simultaneous)
                velocity = random.randint(60, 75)  # Lo-fi is typically more mellow
                for i, note in enumerate(chord_notes):
                    track.append(Message('note_on', note=note, velocity=velocity, time=0 if i > 0 else 0))
                
                # Note off events (after one bar)
                for i, note in enumerate(chord_notes):
                    off_time = bar_duration if i == len(chord_notes) - 1 else 0
                    track.append(Message('note_off', note=note, velocity=0, time=off_time))
        
        # Save MIDI file
        timestamp = int(time.time())
        filename = f"outputs/generic_chords_{timestamp}.mid"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        midi.save(filename)
        
        return filename
    
    def _generate_generic_bass_line(self, key, base_note, bpm):
        """
        Generate a generic bass line based on the key.
        
        Parameters:
        - key: Musical key (e.g., 'C', 'F#')
        - base_note: MIDI base note
        - bpm: Tempo in beats per minute
        
        Returns:
        - Path to generated MIDI file
        """
        import mido
        from mido import Message, MidiFile, MidiTrack
        
        # Create MIDI file
        midi = MidiFile()
        track = MidiTrack()
        midi.tracks.append(track)
        
        # Add tempo
        tempo = mido.bpm2tempo(bpm)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))
        
        # Add track name
        track.append(mido.MetaMessage('track_name', name=f"Lo-Fi Bass in {key}"))
        
        # Define common lo-fi chord progressions relative to the key
        progressions = [
            [0, 3, 4, 3],  # I-IV-V-IV (very common in lo-fi)
            [0, 5, 3, 4],  # I-vi-IV-V
            [5, 3, 0, 4],  # vi-IV-I-V
        ]
        
        # Choose a progression
        progression = random.choice(progressions)
        
        # Major scale steps
        major_scale = [0, 2, 4, 5, 7, 9, 11]
        
        # Calculate diatonic notes based on key
        key_offset = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }
        
        root_offset = key_offset.get(key, 0)
        
        # Duration for one beat
        ticks_per_beat = midi.ticks_per_beat
        beat_duration = ticks_per_beat
        
        # Generate 4-bar progression, repeated twice (32 beats total)
        for _ in range(2):
            for chord_idx in progression:
                # Bass plays primarily the root note across 4 beats
                root = (major_scale[chord_idx] + root_offset) % 12
                note = base_note + root
                
                # Define bass pattern:
                # 80% chance for classic root-fifth pattern
                # 20% chance for more complex pattern
                if random.random() < 0.8:
                    # Root on beats 1 and 3, fifth on beat 2, eighth note on beat 4
                    pattern = [
                        {'note': note, 'duration': beat_duration},  # Beat 1: root
                        {'note': note + 7, 'duration': beat_duration},  # Beat 2: fifth
                        {'note': note, 'duration': beat_duration},  # Beat 3: root
                        {'note': note, 'duration': beat_duration // 2},  # Beat 4.1: root
                        {'note': note + 7, 'duration': beat_duration // 2}   # Beat 4.3: fifth
                    ]
                else:
                    # More varied pattern
                    pattern = [
                        {'note': note, 'duration': beat_duration},  # Beat 1: root
                        {'note': note, 'duration': beat_duration // 2},  # Beat 2.1: root
                        {'note': note + 7, 'duration': beat_duration // 2},  # Beat 2.3: fifth
                        {'note': note, 'duration': beat_duration},  # Beat 3: root
                        {'note': note + 2, 'duration': beat_duration // 2},  # Beat 4.1: second
                        {'note': note + 4, 'duration': beat_duration // 2}   # Beat 4.3: third
                    ]
                
                # Add humanization to velocities
                for note_event in pattern:
                    # Create actual MIDI events
                    velocity = random.randint(70, 90)
                    
                    # Note on
                    track.append(Message('note_on', note=note_event['note'], velocity=velocity, time=0))
                    
                    # Note off
                    track.append(Message('note_off', note=note_event['note'], velocity=0, time=note_event['duration']))
        
        # Save MIDI file
        timestamp = int(time.time())
        filename = f"outputs/generic_bass_{timestamp}.mid"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        midi.save(filename)
        
        return filename
    
    def _key_to_midi_note(self, key):
        """
        Convert a musical key to a MIDI base note (middle C = 60).
        
        Parameters:
        - key: Musical key (e.g., 'C', 'F#')
        
        Returns:
        - MIDI note number for the key
        """
        key_to_semitone = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }
        
        # Default to C if key not recognized
        semitone = key_to_semitone.get(key, 0)
        
        # Return as MIDI note (middle C = 60)
        return 60 + semitone