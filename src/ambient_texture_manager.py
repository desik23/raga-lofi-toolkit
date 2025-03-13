#!/usr/bin/env python3
"""
Ambient Texture Module for Raga Lo-Fi
-------------------------------------
Handles selection and processing of ambient textures that complement
specific ragas based on their mood and characteristics.
"""

import os
import random
import json
import numpy as np
from pydub import AudioSegment
from pydub.effects import low_pass_filter


class AmbientTextureManager:
    """Manages ambient textures that match raga characteristics."""
    
    def __init__(self, textures_path="sample_library/textures"):
        """
        Initialize the ambient texture manager.
        
        Parameters:
        - textures_path: Path to the textures directory
        """
        self.textures_path = textures_path
        self.textures = {
            "peaceful": [],
            "devotional": [],
            "contemplative": [],
            "romantic": [],
            "melancholic": [],
            "morning": [],
            "afternoon": [],
            "evening": [],
            "night": []
        }
        
        # Create directory structure if it doesn't exist
        self._ensure_texture_directories()
        
        # Load texture samples
        self._load_textures()
        
        # Load raga-mood mappings
        self.raga_mappings = self._load_raga_mappings()
    
    def _ensure_texture_directories(self):
        """Create the texture directory structure if it doesn't exist."""
        for mood in self.textures:
            path = os.path.join(self.textures_path, mood)
            os.makedirs(path, exist_ok=True)
    
    def _load_textures(self):
        """Load ambient texture samples from the directories."""
        for mood in self.textures:
            mood_dir = os.path.join(self.textures_path, mood)
            if os.path.exists(mood_dir):
                # Find audio files
                self.textures[mood] = [
                    os.path.join(mood_dir, f) 
                    for f in os.listdir(mood_dir)
                    if f.endswith(('.wav', '.mp3', '.aiff', '.ogg'))
                ]
        
        # Count total textures
        total_textures = sum(len(files) for files in self.textures.values())
        print(f"Loaded {total_textures} ambient textures")
    
    def _load_raga_mappings(self):
        """
        Load mappings between ragas and moods/times.
        If mapping file doesn't exist, create a default one.
        """
        mapping_file = os.path.join(self.textures_path, "raga_mood_mappings.json")
        
        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading mapping file. Creating default mappings.")
        
        # Create default mappings
        default_mappings = {
            # Peaceful ragas
            "bhimpalasi": ["peaceful", "afternoon"],
            "deshkar": ["peaceful", "morning"],
            "durga": ["peaceful", "evening"],
            
            # Devotional ragas
            "bhairav": ["devotional", "morning"],
            "bhairavi": ["devotional", "morning"],
            "todi": ["devotional", "morning"],
            
            # Romantic ragas
            "yaman": ["romantic", "evening"],
            "kedar": ["romantic", "evening"],
            "kamod": ["romantic", "evening"],
            
            # Contemplative ragas
            "marwa": ["contemplative", "evening"],
            "puriya": ["contemplative", "evening"],
            "shree": ["contemplative", "evening"],
            
            # Melancholic ragas
            "malkauns": ["melancholic", "night"],
            "darbari": ["melancholic", "night"],
            "bageshri": ["melancholic", "night"],
            
            # Additional time-based mappings
            "lalit": ["contemplative", "dawn"],
            "ahir_bhairav": ["peaceful", "morning"],
            "shuddh_sarang": ["romantic", "afternoon"],
            "multani": ["contemplative", "afternoon"],
            "puriya_dhanashree": ["contemplative", "evening"],
            "hansadhwani": ["peaceful", "evening"],
            "jaijaiwanti": ["romantic", "night"],
            "chandrakauns": ["melancholic", "night"]
        }
        
        # Save the default mappings
        os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
        with open(mapping_file, 'w') as f:
            json.dump(default_mappings, f, indent=2)
        
        return default_mappings
    
    def select_textures_for_raga(self, raga_id, count=2):
        """
        Select ambient textures that match the characteristics of the raga.
        
        Parameters:
        - raga_id: ID of the raga
        - count: Number of textures to select
        
        Returns:
        - List of selected texture file paths
        """
        selected_textures = []
        
        # Look up the raga in our mappings
        mapped_moods = self.raga_mappings.get(raga_id, [])
        
        # If no mapping exists, determine based on raga name
        if not mapped_moods:
            # Try to guess based on the raga name
            raga_lower = raga_id.lower()
            
            # Simple heuristic matching
            if any(term in raga_lower for term in ["bhairav", "todi", "bhairavi"]):
                mapped_moods = ["devotional", "morning"]
            elif any(term in raga_lower for term in ["yaman", "kamod", "kedar"]):
                mapped_moods = ["romantic", "evening"]
            elif any(term in raga_lower for term in ["malkauns", "darbari", "bageshri"]):
                mapped_moods = ["melancholic", "night"]
            else:
                # Default to peaceful if no match
                mapped_moods = ["peaceful"]
        
        # Create a pool of potential textures
        texture_pool = []
        for mood in mapped_moods:
            if mood in self.textures and self.textures[mood]:
                texture_pool.extend(self.textures[mood])
        
        # If no matching textures found, use generic textures from all categories
        if not texture_pool:
            for mood, files in self.textures.items():
                texture_pool.extend(files)
        
        # Select random textures from the pool
        if texture_pool:
            # Ensure we don't try to select more than exist
            count = min(count, len(texture_pool))
            selected_textures = random.sample(texture_pool, count)
        
        return selected_textures
    
    def process_texture(self, texture_path, target_path=None, duration=60, volume_db=-15):
        """
        Process an ambient texture for lo-fi use (trim, fade, filter).
        
        Parameters:
        - texture_path: Path to the texture file
        - target_path: Output path (None for auto-generated)
        - duration: Target duration in seconds
        - volume_db: Target volume in dB
        
        Returns:
        - Path to the processed texture file
        """
        if not os.path.exists(texture_path):
            print(f"Error: Texture file not found: {texture_path}")
            return None
        
        try:
            # Load audio
            audio = AudioSegment.from_file(texture_path)
            
            # Trim to desired length (loop if too short)
            if len(audio) < duration * 1000:
                # Calculate how many loops we need
                loops_needed = int(np.ceil((duration * 1000) / len(audio)))
                audio = audio * loops_needed
            
            # Trim to exact duration
            audio = audio[:duration * 1000]
            
            # Apply fade in/out
            fade_duration = min(3000, len(audio) // 4)  # 3 seconds or 1/4 of audio, whichever is shorter
            audio = audio.fade_in(fade_duration).fade_out(fade_duration)
            
            # Apply lo-fi processing
            audio = low_pass_filter(audio, 4000)  # Low pass filter at 4kHz
            
            # Adjust volume
            audio = audio.apply_gain(volume_db - audio.dBFS)
            
            # Generate output path if not provided
            if target_path is None:
                filename = os.path.splitext(os.path.basename(texture_path))[0]
                target_path = f"outputs/processed_{filename}.wav"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # Export
            audio.export(target_path, format="wav")
            print(f"Processed texture saved to: {target_path}")
            
            return target_path
            
        except Exception as e:
            print(f"Error processing texture: {e}")
            return None
    
    def add_texture_to_library(self, source_path, mood):
        """
        Add a new texture to the library.
        
        Parameters:
        - source_path: Path to the texture file
        - mood: Mood category for the texture
        
        Returns:
        - Path to the added texture in the library
        """
        if not os.path.exists(source_path):
            print(f"Error: Source file not found: {source_path}")
            return None
        
        if mood not in self.textures:
            print(f"Error: Invalid mood category: {mood}")
            print(f"Valid categories: {list(self.textures.keys())}")
            return None
        
        # Create target directory
        target_dir = os.path.join(self.textures_path, mood)
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy file to library
        import shutil
        target_path = os.path.join(target_dir, os.path.basename(source_path))
        shutil.copy2(source_path, target_path)
        
        # Update in-memory list
        self.textures[mood].append(target_path)
        
        print(f"Added texture to {mood} category: {target_path}")
        return target_path
    
    def get_texture_categories(self):
        """
        Get a list of available texture categories.
        
        Returns:
        - List of category names
        """
        return list(self.textures.keys())