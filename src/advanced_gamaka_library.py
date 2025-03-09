#!/usr/bin/env python3
"""
Advanced Gamaka Library
----------------------
A comprehensive library of traditional Carnatic gamakas (ornamentations)
for authentic raga-based music generation.
"""

import random
import numpy as np
from enum import Enum

class GamakaType(Enum):
    """Enumeration of traditional Carnatic gamaka types."""
    KAMPITA = 0       # Oscillation around a note
    JARU = 1          # Slide between notes
    JANTA = 2         # Doubling of a note
    DHATU = 3         # Stretching a note with emphasis
    ULLASITA = 4      # Quick touch of the next higher note
    AHATA = 5         # Delayed attack of a note
    PRATYAHATA = 6    # Quick double touch of the same note
    TRIPUCHHA = 7     # Three-note ornament
    ANDOLA = 8        # Slow oscillation
    NAMITA = 9        # Bending down from a note
    SPHURITA = 10     # Quick oscillation
    PLAIN = 11        # No gamaka (plain note)

class GamakaIntensity(Enum):
    """Intensity levels for gamakas."""
    NONE = 0.0
    SUBTLE = 0.3
    MODERATE = 0.7
    STRONG = 1.0
    INTENSE = 1.5

class GamakaLibrary:
    """Comprehensive library of gamaka patterns for Carnatic ragas."""
    
    def __init__(self):
        """Initialize the gamaka library."""
        self.raga_gamaka_patterns = {}
        self.setup_raga_gamaka_mappings()
    
    def setup_raga_gamaka_mappings(self):
        """Define which gamakas are characteristic for each raga and on which notes."""
        # Format: {raga_id: {note: [(gamaka_type, probability, intensity), ...], ...}, ...}
        
        # Shankarabharanam (equivalent to major scale)
        self.raga_gamaka_patterns["shankarabharanam"] = {
            0: [(GamakaType.PLAIN, 0.9, GamakaIntensity.NONE)],  # Sa is usually plain
            2: [(GamakaType.JANTA, 0.3, GamakaIntensity.MODERATE),  # Ri
                (GamakaType.PLAIN, 0.7, GamakaIntensity.NONE)],
            4: [(GamakaType.KAMPITA, 0.4, GamakaIntensity.MODERATE),  # Ga
                (GamakaType.PLAIN, 0.6, GamakaIntensity.NONE)],
            5: [(GamakaType.JANTA, 0.3, GamakaIntensity.SUBTLE),  # Ma
                (GamakaType.PLAIN, 0.7, GamakaIntensity.NONE)],
            7: [(GamakaType.DHATU, 0.5, GamakaIntensity.MODERATE),  # Pa
                (GamakaType.PLAIN, 0.5, GamakaIntensity.NONE)],
            9: [(GamakaType.JANTA, 0.3, GamakaIntensity.MODERATE),  # Dha
                (GamakaType.PLAIN, 0.7, GamakaIntensity.NONE)],
            11: [(GamakaType.KAMPITA, 0.5, GamakaIntensity.MODERATE),  # Ni
                 (GamakaType.PLAIN, 0.5, GamakaIntensity.NONE)]
        }
        
        # Mayamalavagowla
        self.raga_gamaka_patterns["mayamalavagowla"] = {
            0: [(GamakaType.PLAIN, 0.9, GamakaIntensity.NONE)],  # Sa
            1: [(GamakaType.KAMPITA, 0.8, GamakaIntensity.STRONG),  # Komal Ri
                (GamakaType.ANDOLA, 0.2, GamakaIntensity.STRONG)],
            4: [(GamakaType.KAMPITA, 0.5, GamakaIntensity.MODERATE),  # Ga
                (GamakaType.PLAIN, 0.5, GamakaIntensity.NONE)],
            5: [(GamakaType.JANTA, 0.4, GamakaIntensity.MODERATE),  # Ma
                (GamakaType.PLAIN, 0.6, GamakaIntensity.NONE)],
            7: [(GamakaType.PLAIN, 0.8, GamakaIntensity.NONE),  # Pa
                (GamakaType.DHATU, 0.2, GamakaIntensity.MODERATE)],
            8: [(GamakaType.KAMPITA, 0.8, GamakaIntensity.STRONG),  # Komal Dha
                (GamakaType.ANDOLA, 0.2, GamakaIntensity.STRONG)],
            11: [(GamakaType.KAMPITA, 0.5, GamakaIntensity.MODERATE),  # Ni
                 (GamakaType.PLAIN, 0.5, GamakaIntensity.NONE)]
        }
        
        # Kalyani
        self.raga_gamaka_patterns["kalyani"] = {
            0: [(GamakaType.PLAIN, 0.9, GamakaIntensity.NONE)],  # Sa
            2: [(GamakaType.JANTA, 0.4, GamakaIntensity.MODERATE),  # Ri
                (GamakaType.PLAIN, 0.6, GamakaIntensity.NONE)],
            4: [(GamakaType.KAMPITA, 0.7, GamakaIntensity.MODERATE),  # Ga
                (GamakaType.ULLASITA, 0.3, GamakaIntensity.MODERATE)],
            6: [(GamakaType.JARU, 0.5, GamakaIntensity.MODERATE),  # Tivra Ma
                (GamakaType.PLAIN, 0.5, GamakaIntensity.NONE)],
            7: [(GamakaType.DHATU, 0.4, GamakaIntensity.MODERATE),  # Pa
                (GamakaType.PLAIN, 0.6, GamakaIntensity.NONE)],
            9: [(GamakaType.JANTA, 0.3, GamakaIntensity.MODERATE),  # Dha
                (GamakaType.PLAIN, 0.7, GamakaIntensity.NONE)],
            11: [(GamakaType.KAMPITA, 0.8, GamakaIntensity.STRONG),  # Ni
                 (GamakaType.SPHURITA, 0.2, GamakaIntensity.STRONG)]
        }
        
        # Hindolam
        self.raga_gamaka_patterns["hindolam"] = {
            0: [(GamakaType.PLAIN, 0.8, GamakaIntensity.NONE),  # Sa
                (GamakaType.DHATU, 0.2, GamakaIntensity.MODERATE)],
            3: [(GamakaType.KAMPITA, 0.7, GamakaIntensity.STRONG),  # Ga
                (GamakaType.ANDOLA, 0.3, GamakaIntensity.STRONG)],
            5: [(GamakaType.JANTA, 0.4, GamakaIntensity.MODERATE),  # Ma
                (GamakaType.PLAIN, 0.6, GamakaIntensity.NONE)],
            8: [(GamakaType.KAMPITA, 0.5, GamakaIntensity.MODERATE),  # Komal Dha
                (GamakaType.PLAIN, 0.5, GamakaIntensity.NONE)],
            10: [(GamakaType.KAMPITA, 0.7, GamakaIntensity.STRONG),  # Komal Ni
                 (GamakaType.ANDOLA, 0.3, GamakaIntensity.STRONG)]
        }
        
        # Darbari Kannada
        self.raga_gamaka_patterns["darbari_kannada"] = {
            0: [(GamakaType.PLAIN, 0.9, GamakaIntensity.NONE)],  # Sa
            1: [(GamakaType.KAMPITA, 0.9, GamakaIntensity.INTENSE),  # Komal Ri
                (GamakaType.ANDOLA, 0.1, GamakaIntensity.INTENSE)],
            3: [(GamakaType.KAMPITA, 0.7, GamakaIntensity.STRONG),  # Komal Ga
                (GamakaType.JARU, 0.3, GamakaIntensity.STRONG)],
            5: [(GamakaType.JANTA, 0.4, GamakaIntensity.MODERATE),  # Ma
                (GamakaType.PLAIN, 0.6, GamakaIntensity.NONE)],
            7: [(GamakaType.PLAIN, 0.7, GamakaIntensity.NONE),  # Pa
                (GamakaType.DHATU, 0.3, GamakaIntensity.MODERATE)],
            8: [(GamakaType.KAMPITA, 0.9, GamakaIntensity.INTENSE),  # Komal Dha
                (GamakaType.ANDOLA, 0.1, GamakaIntensity.INTENSE)],
            10: [(GamakaType.KAMPITA, 0.7, GamakaIntensity.STRONG),  # Komal Ni
                 (GamakaType.ANDOLA, 0.3, GamakaIntensity.STRONG)]
        }
        
        # Keeravani
        self.raga_gamaka_patterns["keeravani"] = {
            0: [(GamakaType.PLAIN, 0.9, GamakaIntensity.NONE)],  # Sa
            2: [(GamakaType.JANTA, 0.4, GamakaIntensity.MODERATE),  # Ri
                (GamakaType.PLAIN, 0.6, GamakaIntensity.NONE)],
            3: [(GamakaType.KAMPITA, 0.6, GamakaIntensity.MODERATE),  # Komal Ga
                (GamakaType.PLAIN, 0.4, GamakaIntensity.NONE)],
            5: [(GamakaType.JANTA, 0.3, GamakaIntensity.MODERATE),  # Ma
                (GamakaType.PLAIN, 0.7, GamakaIntensity.NONE)],
            7: [(GamakaType.PLAIN, 0.7, GamakaIntensity.NONE),  # Pa
                (GamakaType.DHATU, 0.3, GamakaIntensity.MODERATE)],
            9: [(GamakaType.JANTA, 0.4, GamakaIntensity.MODERATE),  # Dha
                (GamakaType.PLAIN, 0.6, GamakaIntensity.NONE)],
            10: [(GamakaType.KAMPITA, 0.6, GamakaIntensity.MODERATE),  # Komal Ni
                 (GamakaType.PLAIN, 0.4, GamakaIntensity.NONE)]
        }
        
        # Add more ragas as needed...
    
    def get_gamaka_for_note(self, raga_id, note_value, user_intensity=1.0):
        """
        Determine the appropriate gamaka for a note in a specific raga.
        
        Parameters:
        - raga_id: ID of the raga
        - note_value: The note value (scale degree)
        - user_intensity: User-defined intensity multiplier (0.0-2.0)
        
        Returns:
        - Tuple of (gamaka_type, intensity_value)
        """
        # Get the note value in the octave (0-11)
        note_in_octave = note_value % 12
        
        # Check if we have specific gamaka patterns for this raga
        if raga_id not in self.raga_gamaka_patterns:
            return (GamakaType.PLAIN, 0.0)  # Default to plain note
        
        # Check if we have specific gamaka patterns for this note in this raga
        if note_in_octave not in self.raga_gamaka_patterns[raga_id]:
            return (GamakaType.PLAIN, 0.0)  # Default to plain note
        
        # Get the available gamaka patterns for this note
        gamaka_options = self.raga_gamaka_patterns[raga_id][note_in_octave]
        
        # Choose a gamaka based on the probability weights
        weights = [option[1] for option in gamaka_options]
        selected_gamaka = random.choices(gamaka_options, weights=weights, k=1)[0]
        
        # Adjust intensity based on user setting
        base_intensity = selected_gamaka[2].value
        adjusted_intensity = base_intensity * user_intensity
        
        return (selected_gamaka[0], adjusted_intensity)
    
    def apply_gamaka_to_note(self, note_event, gamaka_info, next_note=None):
        """
        Apply a specific gamaka to a note event.
        
        Parameters:
        - note_event: Dictionary with note value
        - gamaka_info: Tuple of (gamaka_type, intensity)
        - next_note: The next note in the sequence (for context-aware gamakas)
        
        Returns:
        - Updated note event with appropriate pitch bend data
        """
        gamaka_type, intensity = gamaka_info
        
        # No gamaka needed
        if gamaka_type == GamakaType.PLAIN or intensity < 0.1:
            return {'note': note_event, 'bend': None}
        
        # Create pitch bend based on gamaka type
        if gamaka_type == GamakaType.KAMPITA:
            # Oscillation around the note
            return self._apply_kampita(note_event, intensity)
        
        elif gamaka_type == GamakaType.JARU:
            # Slide between notes
            if next_note is not None:
                return self._apply_jaru(note_event, next_note, intensity)
            else:
                return {'note': note_event, 'bend': None}
        
        elif gamaka_type == GamakaType.JANTA:
            # Double note - handled at the sequence level
            return {'note': note_event, 'bend': None}
        
        elif gamaka_type == GamakaType.DHATU:
            # Stretched note with emphasis
            return self._apply_dhatu(note_event, intensity)
        
        elif gamaka_type == GamakaType.ULLASITA:
            # Quick touch of the next higher note
            return self._apply_ullasita(note_event, intensity)
        
        elif gamaka_type == GamakaType.ANDOLA:
            # Slow oscillation
            return self._apply_andola(note_event, intensity)
        
        elif gamaka_type == GamakaType.NAMITA:
            # Bend down from the note
            return self._apply_namita(note_event, intensity)
        
        elif gamaka_type == GamakaType.SPHURITA:
            # Quick oscillation
            return self._apply_sphurita(note_event, intensity)
            
        else:
            # Default case for other gamakas not specifically implemented
            return {'note': note_event, 'bend': None}
    
    def apply_gamaka_sequence(self, notes, raga_id, user_intensity=1.0):
        """
        Apply appropriate gamakas to a sequence of notes.
        
        Parameters:
        - notes: List of note values
        - raga_id: ID of the raga
        - user_intensity: User-defined intensity multiplier (0.0-2.0)
        
        Returns:
        - List of note events with pitch bend data
        """
        result = []
        
        for i, note in enumerate(notes):
            # Get the next note for context (if available)
            next_note = notes[i+1] if i < len(notes) - 1 else None
            
            # Get appropriate gamaka for this note
            gamaka_info = self.get_gamaka_for_note(raga_id, note, user_intensity)
            
            # Handle special case: Janta (note doubling)
            if gamaka_info[0] == GamakaType.JANTA and random.random() < 0.8:
                # First instance of the note
                first_note = self.apply_gamaka_to_note(note, (GamakaType.PLAIN, 0), next_note)
                result.append(first_note)
                
                # Second instance of the note (with subtle gamaka)
                second_note = self.apply_gamaka_to_note(note, (GamakaType.KAMPITA, gamaka_info[1] * 0.7), next_note)
                result.append(second_note)
            else:
                # Apply the selected gamaka
                note_event = self.apply_gamaka_to_note(note, gamaka_info, next_note)
                result.append(note_event)
        
        return result
    
    # Implementation of specific gamaka types
    
    def _apply_kampita(self, note, intensity):
        """Apply kampita (oscillation) gamaka."""
        bend_points = []
        bend_points.append((0, 0))  # Start at the note
        
        # Intensity affects the depth and number of oscillations
        depth = min(0.5, intensity * 0.3)
        num_cycles = 2 if intensity < 1.0 else 3
        
        # Create multiple oscillation cycles
        for i in range(num_cycles):
            cycle_pos = (i + 1) / (num_cycles + 1)
            pos1 = cycle_pos - 0.1
            pos2 = cycle_pos
            pos3 = cycle_pos + 0.1
            
            # Decrease amplitude slightly for each cycle
            cycle_depth = depth * (1 - (i * 0.2))
            
            bend_points.append((max(0, pos1), cycle_depth))
            bend_points.append((max(0, pos2), -cycle_depth))
            bend_points.append((min(1, pos3), 0))
        
        return {'note': note, 'bend': bend_points}
    
    def _apply_jaru(self, note, next_note, intensity):
        """Apply jaru (slide) gamaka."""
        bend_points = []
        bend_points.append((0, 0))  # Start at current note
        
        # Calculate slide depth based on interval to next note
        interval = next_note - note
        if abs(interval) <= 1:
            # Small interval - subtle slide
            bend_points.append((0.5, -interval * intensity * 0.5))
            bend_points.append((0.9, -interval * intensity * 0.8))
            bend_points.append((1.0, 0))
        else:
            # Larger interval - more pronounced slide
            slide_amount = min(abs(interval) * 0.7, 2.0) * intensity
            if interval > 0:
                # Sliding up
                bend_points.append((0.4, 0))
                bend_points.append((0.7, interval * 0.3 * intensity))
                bend_points.append((1.0, 0))
            else:
                # Sliding down
                bend_points.append((0.4, 0))
                bend_points.append((0.7, -slide_amount))
                bend_points.append((1.0, 0))
        
        return {'note': note, 'bend': bend_points}
    
    def _apply_dhatu(self, note, intensity):
        """Apply dhatu (emphasis) gamaka."""
        bend_points = []
        bend_points.append((0, 0))
        
        # Initial slight dip, then rise above pitch, then settle
        bend_points.append((0.1, -0.1 * intensity))
        bend_points.append((0.3, 0.2 * intensity))
        bend_points.append((0.7, 0.1 * intensity))
        bend_points.append((1.0, 0))
        
        return {'note': note, 'bend': bend_points}
    
    def _apply_ullasita(self, note, intensity):
        """Apply ullasita (quick touch of higher note) gamaka."""
        bend_points = []
        bend_points.append((0, 0))
        
        # Quick rise to touch the next note, then back
        bend_points.append((0.2, 0))
        bend_points.append((0.3, 0.8 * intensity))
        bend_points.append((0.4, 0))
        bend_points.append((1.0, 0))
        
        return {'note': note, 'bend': bend_points}
    
    def _apply_andola(self, note, intensity):
        """Apply andola (slow oscillation) gamaka."""
        bend_points = []
        bend_points.append((0, 0))
        
        # Deeper, slower oscillation compared to kampita
        depth = min(0.7, intensity * 0.5)
        
        # One complete cycle
        bend_points.append((0.25, depth))
        bend_points.append((0.5, 0))
        bend_points.append((0.75, -depth))
        bend_points.append((1.0, 0))
        
        return {'note': note, 'bend': bend_points}
    
    def _apply_namita(self, note, intensity):
        """Apply namita (downward bend) gamaka."""
        bend_points = []
        bend_points.append((0, 0))
        
        # Bend down then back to pitch
        bend_depth = min(1.0, intensity * 0.6)
        bend_points.append((0.3, 0))
        bend_points.append((0.6, -bend_depth))
        bend_points.append((0.9, -bend_depth * 0.5))
        bend_points.append((1.0, 0))
        
        return {'note': note, 'bend': bend_points}
    
    def _apply_sphurita(self, note, intensity):
        """Apply sphurita (quick oscillation) gamaka."""
        bend_points = []
        bend_points.append((0, 0))
        
        # Very quick, tight oscillation
        depth = min(0.4, intensity * 0.25)
        
        # Multiple quick oscillations
        cycles = 4 if intensity > 1.0 else 3
        for i in range(cycles):
            cycle_start = i / cycles
            cycle_mid = (i + 0.5) / cycles
            cycle_end = (i + 1) / cycles
            
            bend_points.append((cycle_start, 0))
            bend_points.append((cycle_start + 0.1/cycles, depth))
            bend_points.append((cycle_mid, -depth))
            bend_points.append((cycle_end - 0.1/cycles, depth))
            
        bend_points.append((1.0, 0))
        
        return {'note': note, 'bend': bend_points}

# Integration with main generator
def integrate_advanced_gamakas(carnatic_features_instance):
    """
    Integrate the advanced gamaka library into the CarnaticFeatures class.
    
    Parameters:
    - carnatic_features_instance: Instance of the CarnaticFeatures class
    
    Returns:
    - Updated carnatic_features_instance with advanced gamaka capabilities
    """
    # Create and attach the gamaka library
    gamaka_lib = GamakaLibrary()
    carnatic_features_instance.gamaka_library = gamaka_lib
    
    # Replace the simple apply_gamaka_with_pitch_bend method with the advanced version
    def enhanced_apply_gamaka_with_pitch_bend(self, notes, raga_id, intensity=1.0):
        """
        Apply advanced characteristic Carnatic gamaka ornamentations to notes.
        
        Parameters:
        - notes: Original note sequence
        - raga_id: ID of the raga
        - intensity: Gamaka intensity multiplier
        
        Returns:
        - List of note events with advanced pitch bend data
        """
        return self.gamaka_library.apply_gamaka_sequence(notes, raga_id, intensity)
    
    # Attach the enhanced method to the class instance
    carnatic_features_instance.apply_gamaka_with_pitch_bend = lambda notes, raga_id, intensity=1.0: \
        enhanced_apply_gamaka_with_pitch_bend(carnatic_features_instance, notes, raga_id, intensity)
    
    return carnatic_features_instance

# Example usage:
if __name__ == "__main__":
    # Test the gamaka library directly
    library = GamakaLibrary()
    
    # Generate a simple phrase
    test_notes = [0, 2, 4, 5, 7, 9, 11, 12, 11, 9, 7, 5, 4, 2, 0]
    
    # Apply gamakas
    test_result = library.apply_gamaka_sequence(test_notes, "kalyani", 1.0)
    
    print("Applied gamakas to a Kalyani phrase. Result:")
    for i, event in enumerate(test_result):
        original_note = test_notes[i]
        has_gamaka = "Yes" if event['bend'] else "No"
        print(f"Note {original_note}: Gamaka = {has_gamaka}")