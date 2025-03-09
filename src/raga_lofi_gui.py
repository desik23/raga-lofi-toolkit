#!/usr/bin/env python3
"""
Raga Lo-Fi Generator GUI
------------------------
A simple graphical interface for the raga-based lo-fi melody generator.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import subprocess
import json
import random
import time  # Add this import
from enhanced_raga_generator import EnhancedRagaGenerator
from raga_identifier import RagaIdentifier
from raga_dataset_manager import RagaDatasetManager

class RagaLoFiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Raga Lo-Fi Generator")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)
        
        # Initialize the generator
        self.generator = EnhancedRagaGenerator()
        
        # Initialize the identifier and dataset manager
        self.identifier = RagaIdentifier()
        self.dataset_manager = RagaDatasetManager()
        
        # Create a top-level notebook
        self.app_notebook = ttk.Notebook(root)
        self.app_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs for generator and identifier
        self.generator_frame = ttk.Frame(self.app_notebook, padding="20")
        self.app_notebook.add(self.generator_frame, text="Raga Generator")
        
        self.identifier_frame = ttk.Frame(self.app_notebook, padding="20")
        self.app_notebook.add(self.identifier_frame, text="Raga Identifier")
        
        # Create status bar (outside the notebook)
        self._create_status_bar()
        
        # Build Generator UI (original functionality)
        self.main_frame = self.generator_frame
        self._create_generator_ui()
        
        # Build Identifier UI (new functionality)
        self._create_identifier_ui()
        
        # Configure styling
        self._configure_styles()
    def _create_generator_ui(self):
        """Create the UI elements for the generator tab."""
        # Create UI elements
        self._create_header()
        self._create_raga_selection()
        self._create_parameters()
        self._create_generation_controls()
        self._create_output_display()
        self._create_notation_display()
        # Set default values
        self._set_defaults()
    def _configure_styles(self):
        """Configure custom styles for the application."""
        style = ttk.Style()
        style.configure("TLabel", font=("Helvetica", 11))
        style.configure("Header.TLabel", font=("Helvetica", 16, "bold"))
        style.configure("Subheader.TLabel", font=("Helvetica", 12, "bold"))
        style.configure("Generate.TButton", font=("Helvetica", 12, "bold"), padding=10)
    
    def _create_header(self):
        """Create the application header."""
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(
            header_frame, 
            text="Raga Lo-Fi Beat Generator", 
            style="Header.TLabel"
        ).pack(side=tk.LEFT)
    
    def _create_raga_selection(self):
        """Create the raga selection section with Carnatic support."""
        raga_frame = ttk.LabelFrame(self.main_frame, text="Raga Selection", padding=10)
        raga_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Add system selection radiobuttons
        system_frame = ttk.Frame(raga_frame)
        system_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(system_frame, text="Music System:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.system_var = tk.StringVar(value="both")
        ttk.Radiobutton(
            system_frame, text="Hindustani", 
            variable=self.system_var, value="hindustani",
            command=self._filter_ragas_by_system
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Radiobutton(
            system_frame, text="Carnatic", 
            variable=self.system_var, value="carnatic",
            command=self._filter_ragas_by_system
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Radiobutton(
            system_frame, text="Both", 
            variable=self.system_var, value="both",
            command=self._filter_ragas_by_system
        ).pack(side=tk.LEFT)
        
        # Selection method tabs
        self.selection_tabs = ttk.Notebook(raga_frame)
        self.selection_tabs.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Direct selection
        direct_frame = ttk.Frame(self.selection_tabs, padding=10)
        self.selection_tabs.add(direct_frame, text="Select Raga")
        
        # Raga selection
        ttk.Label(direct_frame, text="Choose Raga:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        # Get available ragas
        self.available_ragas = self.generator.list_available_ragas()
        self.raga_options = []
        self.raga_ids = []
        
        for raga in self.available_ragas:
            system_tag = ""
            if hasattr(self.generator, 'carnatic') and 'system' in raga:
                system_tag = f" ({raga['system'].capitalize()})"
            
            # Show equivalent raga if available
            equivalent = ""
            if 'hindustani_name' in raga and raga['hindustani_name']:
                equivalent = f" ≈ {raga['hindustani_name']} (Hindustani)"
            elif 'carnatic_name' in raga and raga['carnatic_name']:
                equivalent = f" ≈ {raga['carnatic_name']} (Carnatic)"
                
            self.raga_options.append(f"{raga['name']}{system_tag} - {raga['mood']}{equivalent}")
            self.raga_ids.append(raga['id'])
        
        self.selected_raga = tk.StringVar()
        self.raga_dropdown = ttk.Combobox(
            direct_frame, 
            textvariable=self.selected_raga,
            values=self.raga_options,
            width=60,
            state="readonly"
        )
        self.raga_dropdown.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.raga_dropdown.bind("<<ComboboxSelected>>", self._on_raga_selected)
        
        # Raga description
        self.raga_description = ttk.Label(direct_frame, text="", wraplength=700)
        self.raga_description.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Tab 2: Selection by time
        time_frame = ttk.Frame(self.selection_tabs, padding=10)
        self.selection_tabs.add(time_frame, text="By Time of Day")
        
        # Time selection
        ttk.Label(time_frame, text="Time of Day:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.time_options = ["morning", "afternoon", "evening", "night"]
        self.selected_time = tk.StringVar()
        self.time_dropdown = ttk.Combobox(
            time_frame, 
            textvariable=self.selected_time,
            values=self.time_options,
            width=30,
            state="readonly"
        )
        self.time_dropdown.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.time_dropdown.bind("<<ComboboxSelected>>", self._on_time_selected)
        
        # Ragas for selected time
        ttk.Label(time_frame, text="Available Ragas:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.time_ragas_var = tk.StringVar()
        self.time_ragas_dropdown = ttk.Combobox(
            time_frame, 
            textvariable=self.time_ragas_var,
            values=[],
            width=60,
            state="readonly"
        )
        self.time_ragas_dropdown.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        self.time_ragas_dropdown.bind("<<ComboboxSelected>>", self._on_time_raga_selected)
        
        # Tab 3: Selection by mood
        mood_frame = ttk.Frame(self.selection_tabs, padding=10)
        self.selection_tabs.add(mood_frame, text="By Mood")
        
        # Mood selection
        ttk.Label(mood_frame, text="Mood:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.mood_options = ["focus", "deep_concentration", "relaxation", "creative", "meditative", "devotional"]
        self.selected_mood = tk.StringVar()
        self.mood_dropdown = ttk.Combobox(
            mood_frame, 
            textvariable=self.selected_mood,
            values=self.mood_options,
            width=30,
            state="readonly"
        )
        self.mood_dropdown.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.mood_dropdown.bind("<<ComboboxSelected>>", self._on_mood_selected)
        
        # Ragas for selected mood
        ttk.Label(mood_frame, text="Available Ragas:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.mood_ragas_var = tk.StringVar()
        self.mood_ragas_dropdown = ttk.Combobox(
            mood_frame, 
            textvariable=self.mood_ragas_var,
            values=[],
            width=60,
            state="readonly"
        )
        self.mood_ragas_dropdown.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        self.mood_ragas_dropdown.bind("<<ComboboxSelected>>", self._on_mood_raga_selected)
        
        # Tab 4: Carnatic specific selection (Melakarta)
        if hasattr(self.generator, 'carnatic'):
            melakarta_frame = ttk.Frame(self.selection_tabs, padding=10)
            self.selection_tabs.add(melakarta_frame, text="Melakarta")
            
            # Melakarta selection
            ttk.Label(melakarta_frame, text="Melakarta Parent:").grid(row=0, column=0, sticky=tk.W, pady=5)
            
            # Get all unique melakarta numbers
            melakarta_numbers = sorted(list(set(
                raga.get('melakarta') for raga in self.generator.ragas_data.values() 
                if raga.get('melakarta') is not None
            )))
            
            # Create option list with names if available
            melakarta_options = []
            for num in melakarta_numbers:
                # Find a raga with this melakarta number to get its name
                for raga in self.generator.ragas_data.values():
                    if raga.get('melakarta') == num:
                        melakarta_options.append(f"{num}: {raga['name']}")
                        break
                else:
                    melakarta_options.append(str(num))
            
            self.melakarta_var = tk.StringVar()
            self.melakarta_dropdown = ttk.Combobox(
                melakarta_frame, 
                textvariable=self.melakarta_var,
                values=melakarta_options,
                width=40,
                state="readonly"
            )
            self.melakarta_dropdown.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
            self.melakarta_dropdown.bind("<<ComboboxSelected>>", self._on_melakarta_selected)
            
            # Ragas for selected melakarta
            ttk.Label(melakarta_frame, text="Available Ragas:").grid(row=1, column=0, sticky=tk.W, pady=5)
            self.melakarta_ragas_var = tk.StringVar()
            self.melakarta_ragas_dropdown = ttk.Combobox(
                melakarta_frame, 
                textvariable=self.melakarta_ragas_var,
                values=[],
                width=60,
                state="readonly"
            )
            self.melakarta_ragas_dropdown.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
            self.melakarta_ragas_dropdown.bind("<<ComboboxSelected>>", self._on_melakarta_raga_selected)
        
    def _create_parameters(self):
        """Create parameter controls with Carnatic enhancements."""
        params_frame = ttk.LabelFrame(self.main_frame, text="Generation Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Create three columns
        left_frame = ttk.Frame(params_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        middle_frame = ttk.Frame(params_frame)
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_frame = ttk.Frame(params_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Left column - BPM
        bpm_frame = ttk.Frame(left_frame)
        bpm_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(bpm_frame, text="Tempo (BPM):").pack(side=tk.LEFT)
        
        self.bpm_var = tk.IntVar()
        self.bpm_scale = ttk.Scale(
            bpm_frame, 
            from_=60, 
            to=90, 
            orient=tk.HORIZONTAL, 
            variable=self.bpm_var,
            command=self._update_bpm_label
        )
        self.bpm_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        self.bpm_label = ttk.Label(bpm_frame, text="75")
        self.bpm_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Left column - Use patterns
        pattern_frame = ttk.Frame(left_frame)
        pattern_frame.pack(fill=tk.X, pady=5)
        
        self.use_patterns_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            pattern_frame, 
            text="Use characteristic patterns", 
            variable=self.use_patterns_var
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        # Left column - Traditional rules toggle
        self.strict_rules_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            left_frame, 
            text="Follow traditional raga rules strictly", 
            variable=self.strict_rules_var
        ).pack(anchor=tk.W, pady=5)
        
        # Middle column - Gamaka intensity
        gamaka_frame = ttk.Frame(middle_frame)
        gamaka_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(gamaka_frame, text="Gamaka Intensity:").pack(side=tk.LEFT)
        
        self.gamaka_var = tk.DoubleVar(value=1.0)
        self.gamaka_scale = ttk.Scale(
            gamaka_frame, 
            from_=0.0, 
            to=2.0, 
            orient=tk.HORIZONTAL, 
            variable=self.gamaka_var,
            command=self._update_gamaka_label
        )
        self.gamaka_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        self.gamaka_label = ttk.Label(gamaka_frame, text="Normal (1.0)")
        self.gamaka_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Middle column - Melody length
        melody_length_frame = ttk.Frame(middle_frame)
        melody_length_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(melody_length_frame, text="Melody length:").pack(side=tk.LEFT)
        
        self.melody_length_var = tk.IntVar(value=32)
        self.melody_length_dropdown = ttk.Combobox(
            melody_length_frame, 
            textvariable=self.melody_length_var,
            values=[16, 24, 32, 48, 64],
            width=5,
            state="readonly"
        )
        self.melody_length_dropdown.pack(side=tk.LEFT, padx=5)
        ttk.Label(melody_length_frame, text="notes").pack(side=tk.LEFT)
        
        # Right column - Base note
        base_note_frame = ttk.Frame(right_frame)
        base_note_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(base_note_frame, text="Base note:").pack(side=tk.LEFT)
        
        self.base_notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        self.base_note_var = tk.StringVar(value="C")
        self.base_note_dropdown = ttk.Combobox(
            base_note_frame, 
            textvariable=self.base_note_var,
            values=self.base_notes,
            width=5,
            state="readonly"
        )
        self.base_note_dropdown.pack(side=tk.LEFT, padx=5)
        
        self.octave_var = tk.IntVar(value=4)
        self.octave_dropdown = ttk.Combobox(
            base_note_frame, 
            textvariable=self.octave_var,
            values=[3, 4, 5],
            width=3,
            state="readonly"
        )
        self.octave_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Right column - Notation display preference
        notation_frame = ttk.Frame(right_frame)
        notation_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(notation_frame, text="Show notation:").pack(side=tk.LEFT)
        
        self.notation_var = tk.StringVar(value="western")
        notation_dropdown = ttk.Combobox(
            notation_frame,
            textvariable=self.notation_var,
            values=["western", "hindustani", "carnatic", "all"],
            width=12,
            state="readonly"
        )
        notation_dropdown.pack(side=tk.LEFT, padx=5)


    def _update_gamaka_label(self, value):
        """Update Gamaka intensity label when slider is moved."""
        value = float(value)
        if value < 0.1:
            label = "None (0.0)"
        elif value < 0.5:
            label = "Light ({:.1f})".format(value)
        elif value < 0.9:
            label = "Medium ({:.1f})".format(value)
        elif value < 1.5:
            label = "Normal ({:.1f})".format(value)
        else:
            label = "Heavy ({:.1f})".format(value)
            
        self.gamaka_label.config(text=label)


    def _create_notation_display(self):
        """Create a section to display raga notes in different notation systems."""
        notation_frame = ttk.LabelFrame(self.main_frame, text="Raga Notation", padding=10)
        notation_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Create a Text widget to display the notation
        self.notation_text = tk.Text(notation_frame, wrap=tk.WORD, height=3)
        self.notation_text.pack(fill=tk.X)
        
        # Configure tags for formatting
        self.notation_text.tag_configure("title", font=("Helvetica", 11, "bold"))
        self.notation_text.tag_configure("notation", font=("Courier", 10))
        
        # Make read-only
        self.notation_text.config(state=tk.DISABLED)


    def _update_notation_display(self, raga_id):
        """Update the notation display based on selected raga and notation preference."""
        if raga_id not in self.generator.ragas_data:
            return
            
        raga = self.generator.ragas_data[raga_id]
        notation_pref = self.notation_var.get()
        
        # Enable text widget for editing
        self.notation_text.config(state=tk.NORMAL)
        
        # Clear previous content
        self.notation_text.delete(1.0, tk.END)
        
        # Insert raga name
        self.notation_text.insert(tk.END, f"{raga['name']} - ", "title")
        
        # Get scale degrees
        arohan = raga.get('arohan', [])
        avarohan = raga.get('avarohan', [])
        
        # Convert to different notation systems
        western = self._get_western_notation(arohan, avarohan)
        hindustani = self._get_hindustani_notation(arohan, avarohan)
        carnatic = self._get_carnatic_notation(arohan, avarohan)
        
        # Display based on preference
        if notation_pref == "western" or notation_pref == "all":
            self.notation_text.insert(tk.END, "Western: ", "title")
            self.notation_text.insert(tk.END, western + "\n", "notation")
            
        if notation_pref == "hindustani" or notation_pref == "all":
            self.notation_text.insert(tk.END, "Hindustani: ", "title")
            self.notation_text.insert(tk.END, hindustani + "\n", "notation")
            
        if notation_pref == "carnatic" or notation_pref == "all":
            self.notation_text.insert(tk.END, "Carnatic: ", "title")
            self.notation_text.insert(tk.END, carnatic, "notation")
        
        # Make read-only again
        self.notation_text.config(state=tk.DISABLED)


    def _get_western_notation(self, arohan, avarohan):
        """Convert scale degrees to Western notation."""
        western_notes = ["C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"]
        
        # Get base note
        base_idx = self.base_notes.index(self.base_note_var.get())
        
        # Convert arohan
        arohan_notes = []
        for degree in arohan:
            note_idx = (base_idx + degree) % 12
            arohan_notes.append(western_notes[note_idx])
        
        # Convert avarohan
        avarohan_notes = []
        for degree in avarohan:
            note_idx = (base_idx + degree) % 12
            avarohan_notes.append(western_notes[note_idx])
        
        return "↑ " + " ".join(arohan_notes) + " | ↓ " + " ".join(avarohan_notes)


    def _get_hindustani_notation(self, arohan, avarohan):
        """Convert scale degrees to Hindustani notation."""
        # Basic mapping
        hindustani_notes = {
            0: "Sa",
            1: "re",
            2: "Re",
            3: "ga",
            4: "Ga",
            5: "ma",
            6: "Ma",
            7: "Pa",
            8: "dha",
            9: "Dha",
            10: "ni",
            11: "Ni",
            12: "Sa'"
        }
        
        # Convert arohan
        arohan_notes = [hindustani_notes.get(degree, str(degree)) for degree in arohan]
        
        # Convert avarohan
        avarohan_notes = [hindustani_notes.get(degree, str(degree)) for degree in avarohan]
        
        return "↑ " + " ".join(arohan_notes) + " | ↓ " + " ".join(avarohan_notes)


    def _get_carnatic_notation(self, arohan, avarohan):
        """Convert scale degrees to Carnatic notation."""
        # Carnatic notation is more complex with multiple variants of each note
        carnatic_notes = {
            0: "Sa",
            1: "Ri₁",
            2: "Ri₂",
            3: "Ga₁",
            4: "Ga₂",
            5: "Ma₁",
            6: "Ma₂",
            7: "Pa",
            8: "Dha₁",
            9: "Dha₂",
            10: "Ni₁",
            11: "Ni₂",
            12: "Sa'"
        }
        
        # Convert arohan
        arohan_notes = [carnatic_notes.get(degree, str(degree)) for degree in arohan]
        
        # Convert avarohan
        avarohan_notes = [carnatic_notes.get(degree, str(degree)) for degree in avarohan]
        
        return "↑ " + " ".join(arohan_notes) + " | ↓ " + " ".join(avarohan_notes)


    def _on_raga_selected(self, event):
        """Update display when raga is selected, with notation display."""
        if not self.selected_raga.get():
            return
            
        # Find the selected raga (existing code...)
        idx = self.raga_dropdown.current()
        if idx >= 0:
            # Use filtered IDs if filtering by system
            if hasattr(self, 'filtered_raga_ids') and self.filtered_raga_ids:
                selected_id = self.filtered_raga_ids[idx]
            else:
                selected_id = self.raga_ids[idx]
                
            raga_data = self.generator.ragas_data.get(selected_id, {})
            
            # Update description (existing code...)
            description = f"System: {raga_data.get('system', 'hindustani').capitalize()}\n"
            
            # Show equivalent if available
            if raga_data.get('hindustani_equivalent'):
                h_name = raga_data.get('hindustani_name', raga_data.get('hindustani_equivalent'))
                description += f"Hindustani Equivalent: {h_name}\n"
            elif raga_data.get('carnatic_equivalent'):
                c_name = raga_data.get('carnatic_name', raga_data.get('carnatic_equivalent'))
                description += f"Carnatic Equivalent: {c_name}\n"
                
            # Show melakarta for Carnatic ragas
            if raga_data.get('melakarta'):
                description += f"Melakarta: {raga_data.get('melakarta')}\n"
                
            description += f"Time: {raga_data.get('time', 'N/A')}\n"
            description += f"Mood: {raga_data.get('mood', 'N/A')}\n"
            description += f"Suitable for: {', '.join(raga_data.get('suitable_for', ['N/A']))}"
            
            self.raga_description.config(text=description)
            
            # Update notation display
            self._update_notation_display(selected_id)



    
    def _create_generation_controls(self):
        """Create generation control buttons."""
        controls_frame = ttk.Frame(self.main_frame)
        controls_frame.pack(fill=tk.X, pady=15)
        
        # Generation options
        options_frame = ttk.LabelFrame(controls_frame, text="Generate", padding=10)
        options_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        # Component options
        components_frame = ttk.Frame(options_frame)
        components_frame.pack(fill=tk.X, pady=5)
        
        self.melody_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            components_frame, 
            text="Melody", 
            variable=self.melody_var
        ).pack(side=tk.LEFT, padx=5)
        
        self.chords_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            components_frame, 
            text="Chords", 
            variable=self.chords_var
        ).pack(side=tk.LEFT, padx=5)
        
        self.bass_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            components_frame, 
            text="Bass", 
            variable=self.bass_var
        ).pack(side=tk.LEFT, padx=5)
        
        # Generate button
        generate_button = ttk.Button(
            options_frame,
            text="Generate MIDI Files",
            command=self._generate_files,
            style="Generate.TButton"
        )
        generate_button.pack(fill=tk.X, pady=10)
        
        # Actions frame
        actions_frame = ttk.LabelFrame(controls_frame, text="Actions", padding=10)
        actions_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Buttons
        buttons_frame = ttk.Frame(actions_frame)
        buttons_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            buttons_frame,
            text="Open Output Folder",
            command=self._open_output_folder
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            buttons_frame,
            text="Open in Logic Pro X",
            command=self._open_in_logic
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            buttons_frame,
            text="Save Settings",
            command=self._save_settings
        ).pack(side=tk.LEFT, padx=5)
    
    def _create_output_display(self):
        """Create output display area."""
        output_frame = ttk.LabelFrame(self.main_frame, text="Generated Files", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a Text widget with scrollbar
        self.output_text = tk.Text(output_frame, wrap=tk.WORD, height=10)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(output_frame, command=self.output_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.output_text.config(yscrollcommand=scrollbar.set)
        self.output_text.tag_configure("title", font=("Helvetica", 11, "bold"))
        self.output_text.tag_configure("path", font=("Courier", 10))
        
        # Make read-only
        self.output_text.config(state=tk.DISABLED)
    
    def _create_status_bar(self):
        """Create status bar at the bottom."""
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _set_defaults(self):
        """Set default values for all controls."""
        if self.raga_options:
            self.raga_dropdown.current(0)
            self._on_raga_selected(None)
        
        self.bpm_var.set(75)
        self._update_bpm_label(75)
        
        self.use_patterns_var.set(True)
        
        self.base_note_var.set("C")
        self.octave_var.set(4)
        
        self.melody_length_var.set(32)
        
        if self.time_options:
            self.time_dropdown.current(0)
            self._on_time_selected(None)
        
        if self.mood_options:
            self.mood_dropdown.current(0)
            self._on_mood_selected(None)
    
    def _update_bpm_label(self, value):
        """Update BPM label when slider is moved."""
        self.bpm_label.config(text=str(int(float(value))))
    

    
    def _on_time_selected(self, event):
        """Update ragas list when time is selected."""
        selected_time = self.selected_time.get()
        if not selected_time:
            return
        
        # Get ragas for the selected time
        raga_ids = self.generator.get_ragas_by_time(selected_time)
        
        # Update dropdown
        if raga_ids:
            raga_options = []
            for raga_id in raga_ids:
                if raga_id in self.generator.ragas_data:
                    raga = self.generator.ragas_data[raga_id]
                    raga_options.append(f"{raga['name']} - {raga['mood']}")
            
            self.time_ragas_dropdown.config(values=raga_options)
            self.time_ragas_dropdown.current(0)
            self.time_ragas_var.set(raga_options[0])
            
            # Store the raga ids for reference
            self.time_raga_ids = raga_ids
        else:
            self.time_ragas_dropdown.config(values=["No ragas available"])
            self.time_ragas_dropdown.current(0)
            self.time_raga_ids = []
    
    def _on_time_raga_selected(self, event):
        """Handle selection of raga from time-based tab."""
        if not hasattr(self, 'time_raga_ids') or not self.time_raga_ids:
            return
            
        idx = self.time_ragas_dropdown.current()
        if idx >= 0 and idx < len(self.time_raga_ids):
            # Find the matching raga in the main dropdown
            selected_id = self.time_raga_ids[idx]
            if selected_id in self.raga_ids:
                main_idx = self.raga_ids.index(selected_id)
                self.raga_dropdown.current(main_idx)
                self.selected_raga.set(self.raga_options[main_idx])
                self._on_raga_selected(None)
                
                # Switch to the main tab
                self.selection_tabs.select(0)
    
    def _on_mood_selected(self, event):
        """Update ragas list when mood is selected."""
        selected_mood = self.selected_mood.get()
        if not selected_mood:
            return
        
        # Get ragas for the selected mood
        raga_ids = self.generator.get_ragas_by_mood(selected_mood)
        
        # Update dropdown
        if raga_ids:
            raga_options = []
            for raga_id in raga_ids:
                if raga_id in self.generator.ragas_data:
                    raga = self.generator.ragas_data[raga_id]
                    raga_options.append(f"{raga['name']} - {raga['mood']}")
            
            self.mood_ragas_dropdown.config(values=raga_options)
            self.mood_ragas_dropdown.current(0)
            self.mood_ragas_var.set(raga_options[0])
            
            # Store the raga ids for reference
            self.mood_raga_ids = raga_ids
        else:
            self.mood_ragas_dropdown.config(values=["No ragas available"])
            self.mood_ragas_dropdown.current(0)
            self.mood_raga_ids = []
    
    def _on_mood_raga_selected(self, event):
        """Handle selection of raga from mood-based tab."""
        if not hasattr(self, 'mood_raga_ids') or not self.mood_raga_ids:
            return
            
        idx = self.mood_ragas_dropdown.current()
        if idx >= 0 and idx < len(self.mood_raga_ids):
            # Find the matching raga in the main dropdown
            selected_id = self.mood_raga_ids[idx]
            if selected_id in self.raga_ids:
                main_idx = self.raga_ids.index(selected_id)
                self.raga_dropdown.current(main_idx)
                self.selected_raga.set(self.raga_options[main_idx])
                self._on_raga_selected(None)
                
                # Switch to the main tab
                self.selection_tabs.select(0)
    def _filter_ragas_by_system(self):
        """Filter ragas based on selected music system."""
        system = self.system_var.get()
        
        if not hasattr(self.generator, 'carnatic'):
            return
        
        # Filter ragas
        filtered_ragas = []
        filtered_ids = []
        
        for i, raga in enumerate(self.available_ragas):
            raga_id = self.raga_ids[i]
            raga_data = self.generator.ragas_data.get(raga_id, {})
            raga_system = raga_data.get('system', 'hindustani')  # Default to hindustani for backward compatibility
            
            if system == 'both' or system == raga_system:
                filtered_ragas.append(self.raga_options[i])
                filtered_ids.append(raga_id)
        
        # Update dropdown
        self.raga_dropdown.config(values=filtered_ragas)
        # Store the filtered IDs
        self.filtered_raga_ids = filtered_ids
        
        # Reset selection if needed
        if self.raga_dropdown.get() not in filtered_ragas:
            if filtered_ragas:
                self.raga_dropdown.current(0)
                self._on_raga_selected(None)
            else:
                self.selected_raga.set('')
                self.raga_description.config(text="No ragas available for selected system")


    def _on_melakarta_selected(self, event):
        """Update ragas list when melakarta is selected."""
        if not hasattr(self.generator, 'carnatic') or not self.melakarta_var.get():
            return
            
        # Extract melakarta number from selection
        melakarta_str = self.melakarta_var.get()
        melakarta_num = int(melakarta_str.split(':')[0])
        
        # Get ragas for this melakarta
        raga_ids = self.generator.carnatic.get_ragas_by_melakarta(melakarta_num)
        
        # Update dropdown
        if raga_ids:
            raga_options = []
            for raga_id in raga_ids:
                if raga_id in self.generator.ragas_data:
                    raga = self.generator.ragas_data[raga_id]
                    equivalent = ""
                    if raga.get('hindustani_name'):
                        equivalent = f" ≈ {raga['hindustani_name']} (Hindustani)"
                    raga_options.append(f"{raga['name']} - {raga['mood']}{equivalent}")
            
            self.melakarta_ragas_dropdown.config(values=raga_options)
            self.melakarta_ragas_dropdown.current(0)
            self.melakarta_ragas_var.set(raga_options[0])
            
            # Store the raga ids for reference
            self.melakarta_raga_ids = raga_ids
        else:
            self.melakarta_ragas_dropdown.config(values=["No ragas available"])
            self.melakarta_ragas_dropdown.current(0)
            self.melakarta_raga_ids = []


    def _on_melakarta_raga_selected(self, event):
        """Handle selection of raga from melakarta tab."""
        if not hasattr(self, 'melakarta_raga_ids') or not self.melakarta_raga_ids:
            return
            
        idx = self.melakarta_ragas_dropdown.current()
        if idx >= 0 and idx < len(self.melakarta_raga_ids):
            # Find the matching raga in the main dropdown
            selected_id = self.melakarta_raga_ids[idx]
            
            # Check if we need to update system filter
            system = self.system_var.get()
            if system != 'both' and system != 'carnatic':
                self.system_var.set('carnatic')
                self._filter_ragas_by_system()
            
            # Find the index in filtered list
            if hasattr(self, 'filtered_raga_ids') and selected_id in self.filtered_raga_ids:
                filtered_idx = self.filtered_raga_ids.index(selected_id)
                self.raga_dropdown.current(filtered_idx)
                self._on_raga_selected(None)
            
            # Switch to the main tab
            self.selection_tabs.select(0)
    def _generate_thread(self, raga_id, base_note, bpm, use_patterns, melody_length):
        """Thread function for file generation with gamaka support."""
        try:
            generated_files = {}
            
            # Get additional parameters
            gamaka_intensity = self.gamaka_var.get()
            strict_rules = self.strict_rules_var.get()
            
            # Generate selected components
            if self.melody_var.get():
                melody_file = self.generator.generate_melody(
                    raga_id, 
                    length=melody_length,
                    use_patterns=use_patterns,
                    base_note=base_note,
                    bpm=bpm,
                    gamaka_intensity=gamaka_intensity,
                    strict_rules=strict_rules
                )
                generated_files['Melody'] = melody_file
            
            if self.chords_var.get():
                chord_file = self.generator.generate_chord_progression(
                    raga_id,
                    length=4,
                    base_note=base_note-12,
                    bpm=bpm,
                    strict_rules=strict_rules
                )
                generated_files['Chords'] = chord_file
            
            if self.bass_var.get():
                bass_file = self.generator.generate_bass_line(
                    raga_id,
                    length=melody_length,
                    base_note=base_note-24,
                    bpm=bpm
                )
                generated_files['Bass'] = bass_file
            
            # Update UI in the main thread
            self.root.after(0, lambda: self._update_output_display(
                self.generator.ragas_data[raga_id]['name'],
                generated_files
            ))
            
            # Update status in the main thread
            self.root.after(0, lambda: self.status_var.set("Generation complete"))
            
        except Exception as e:
            # Show error in the main thread
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, lambda: self.status_var.set("Error generating files"))

    def _generate_files(self):
        """Generate MIDI files based on current settings."""
        # Get selected raga
        idx = self.raga_dropdown.current()
        if idx < 0:
            messagebox.showerror("Error", "Please select a raga")
            return
            
        selected_id = self.raga_ids[idx]
        
        # Get parameters
        bpm = self.bpm_var.get()
        use_patterns = self.use_patterns_var.get()
        
        # Calculate base note
        base_note_idx = self.base_notes.index(self.base_note_var.get())
        octave = self.octave_var.get()
        base_note = 12 * (octave + 1) + base_note_idx
        
        melody_length = self.melody_length_var.get()
        
        # Update status
        self.status_var.set(f"Generating files for {self.generator.ragas_data[selected_id]['name']}...")
        
        # Run generation in a separate thread to keep UI responsive
        threading.Thread(
            target=self._generate_thread, 
            args=(selected_id, base_note, bpm, use_patterns, melody_length)
        ).start()
    def _create_identifier_ui(self):
        """Create the UI elements for the raga identifier tab."""
        # File selection section
        file_section = ttk.LabelFrame(self.identifier_frame, text="Audio File", padding=10)
        file_section.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Button(file_section, text="Load Audio File", command=self._load_audio_file).pack(side=tk.LEFT, padx=(0, 10))
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_section, textvariable=self.file_path_var, width=40, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Identification results section
        results_section = ttk.LabelFrame(self.identifier_frame, text="Identification Results", padding=10)
        results_section.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        self.results_text = tk.Text(results_section, wrap=tk.WORD, height=10)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(results_section, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        
        # Make read-only
        self.results_text.config(state=tk.DISABLED)
        
        # Feedback section
        feedback_section = ttk.LabelFrame(self.identifier_frame, text="Provide Feedback", padding=10)
        feedback_section.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(feedback_section, text="Correct Raga:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.feedback_raga_var = tk.StringVar()
        self.feedback_raga_dropdown = ttk.Combobox(feedback_section, textvariable=self.feedback_raga_var, state="readonly")
        self.feedback_raga_dropdown.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(feedback_section, text="Submit Feedback", command=self._submit_feedback).pack(side=tk.LEFT)
        # Inside your _create_identifier_ui method, add this after the feedback section:
        ttk.Button(feedback_section, text="Populate Dropdown", command=self._populate_raga_dropdown).pack(side=tk.LEFT, padx=10)
        # Performance metrics section
        metrics_section = ttk.LabelFrame(self.identifier_frame, text="Performance Metrics", padding=10)
        metrics_section.pack(fill=tk.X)
        
        # Add buttons to evaluate accuracy and generate report
        buttons_frame = ttk.Frame(metrics_section)
        buttons_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(buttons_frame, text="Evaluate Accuracy", command=self._evaluate_accuracy).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Generate Report", command=self._generate_report).pack(side=tk.LEFT, padx=5)
        
        # Metrics display
        self.metrics_text = tk.Text(metrics_section, wrap=tk.WORD, height=5)
        self.metrics_text.pack(fill=tk.X, expand=True)
        
        # Make read-only
        self.metrics_text.config(state=tk.DISABLED)
    def _load_audio_file(self):
        """Load an audio file for identification."""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=(("Audio Files", "*.wav *.mp3 *.ogg *.flac"), ("All files", "*.*"))
        )
        
        if not file_path:
            return
        
        self.file_path_var.set(file_path)
        self.status_var.set(f"Identifying raga in {os.path.basename(file_path)}...")
        
        # Enable and clear results text
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Analyzing {os.path.basename(file_path)}...\n")
        self.results_text.config(state=tk.DISABLED)
        
        # Run identification in a separate thread to keep UI responsive
        threading.Thread(
            target=self._identify_raga_thread,
            args=(file_path,)
        ).start()

    def _identify_raga_thread(self, file_path):
        """Thread function for raga identification."""
        try:
            # Update status
            self.root.after(0, lambda: self._update_text_status("Processing audio..."))
            
            # Identify raga
            results = self.identifier.identify_raga(file_path, preprocess=True)
            
            # Update UI in the main thread
            if results:
                self.root.after(0, lambda: self._update_identification_results(results))
            else:
                self.root.after(0, lambda: self._update_text_status("No results found. Check console for errors."))
                
        except Exception as e:
            error_msg = f"Error identifying raga: {str(e)}"
            self.root.after(0, lambda: self._update_text_status(error_msg))
            self.root.after(0, lambda: self.status_var.set("Error identifying raga"))
            import traceback
            traceback.print_exc()

    def _update_text_status(self, message):
        """Update the results text with a status message."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.insert(tk.END, f"\n{message}\n")
        self.results_text.see(tk.END)
        self.results_text.config(state=tk.DISABLED)

    def _update_identification_results(self, results):
        """Update the identification results display."""
        # Enable text widget for editing
        self.results_text.config(state=tk.NORMAL)
        
        # Clear previous content
        self.results_text.delete(1.0, tk.END)
        
        # Add identification results
        if not results:
            self.results_text.insert(tk.END, "No raga matches found.\n\n")
            self.results_text.insert(tk.END, "The audio analysis did not find a clear match with known ragas.\n")
            self.results_text.insert(tk.END, "You can still provide feedback below to help improve the system.")
        else:
            # Try to use the get_description method
            try:
                description = self.identifier.get_description()
                self.results_text.insert(tk.END, description)
            except Exception as e:
                # Fallback to basic display
                self.results_text.insert(tk.END, f"Raga Analysis Results:\n\n")
                
                # Display matches if available
                if ('overall_results' in results and 
                    'top_matches' in results['overall_results'] and 
                    results['overall_results']['top_matches']):
                    
                    top_matches = results['overall_results']['top_matches']
                    for i, match in enumerate(top_matches):
                        raga_id = match.get('raga_id', 'Unknown')
                        raga_name = match.get('raga_name', 'Unknown')
                        confidence = match.get('confidence', 0.0)
                        
                        self.results_text.insert(tk.END, f"{i+1}. {raga_name} ({raga_id}) - Confidence: {confidence:.2f}\n")
                else:
                    self.results_text.insert(tk.END, "No clear raga matches found.\n")
                    self.results_text.insert(tk.END, "Please provide feedback below to help improve the system.")
        
        # Make read-only again
        self.results_text.config(state=tk.DISABLED)
        
        # Update status
        self.status_var.set("Raga identification complete")
        
        # Populate the feedback dropdown with all available ragas
        self._populate_raga_dropdown()

    def _populate_raga_dropdown(self):
        """Populate the feedback dropdown with all available ragas without bias."""
        all_ragas = []
        
        # Get all available raga models
        if hasattr(self.identifier, 'feature_extractor') and hasattr(self.identifier.feature_extractor, 'raga_models'):
            for raga_id, model in self.identifier.feature_extractor.raga_models.items():
                raga_name = "Unknown"
                if isinstance(model, dict):
                    if 'metadata' in model and 'raga_name' in model['metadata']:
                        raga_name = model['metadata']['raga_name']
                    elif 'name' in model:
                        raga_name = model['name']
                    else:
                        raga_name = raga_id.capitalize()
                
                all_ragas.append(f"{raga_name} ({raga_id})")
        
        # If no ragas in models, provide a way to add a new one
        if not all_ragas:
            all_ragas.append("New Raga (add_new)")
        
        # Sort alphabetically
        all_ragas.sort()
        
        # Update the dropdown values
        self.feedback_raga_dropdown.config(values=all_ragas)
        
        # Don't auto-select anything - let the user choose
        if not self.feedback_raga_var.get() and all_ragas:
            # Only set a default value if there are actual ragas
            self.feedback_raga_var.set("")
            self.feedback_raga_dropdown.set("")
            
    def _submit_feedback(self):
        """Submit feedback on the identified raga."""
        feedback_raga = self.feedback_raga_var.get()
        
        if not feedback_raga:
            messagebox.showerror("Error", "Please select the correct raga")
            return
        
        # Extract raga ID and name from the selected value
        # Format: "Raga Name (raga_id)"
        try:
            raga_parts = feedback_raga.rsplit(" (", 1)  # Split from the last occurrence
            raga_name = raga_parts[0]
            raga_id = raga_parts[1].rstrip(")")
        except IndexError:
            # Fallback if the format is different
            raga_id = feedback_raga
            raga_name = feedback_raga
        
        # Add the file to the dataset with the correct raga ID
        file_path = self.file_path_var.get()
        
        if not file_path:
            messagebox.showerror("Error", "No audio file loaded")
            return
        
        # Verify dataset manager
        if not hasattr(self, 'dataset_manager'):
            self.dataset_manager = RagaDatasetManager()
        
        # Update text status
        self._update_text_status(f"Submitting feedback: {raga_name} ({raga_id})")
        
        # Add file to dataset
        try:
            result = self.dataset_manager.add_file(file_path, raga_id, raga_name)
            
            if result:
                # Provide feedback to the identifier
                if hasattr(self.identifier, 'provide_feedback'):
                    try:
                        success = self.identifier.provide_feedback(
                            self.identifier.analysis_results or {},
                            raga_id,
                            raga_name,
                            update_model=True
                        )
                        
                        if success:
                            messagebox.showinfo("Success", "Feedback submitted and model updated")
                            self.status_var.set("Feedback submitted and model updated")
                            
                            # Update performance metrics
                            self._evaluate_accuracy()
                        else:
                            messagebox.showwarning("Warning", "Feedback submitted but model update may have failed")
                            self.status_var.set("Feedback submitted but model update may have failed")
                    except Exception as e:
                        messagebox.showwarning("Warning", f"Feedback submitted but error during model update: {str(e)}")
                        self.status_var.set("Feedback submitted but error during model update")
                else:
                    messagebox.showinfo("Success", "Feedback submitted")
                    self.status_var.set("Feedback submitted")
            else:
                messagebox.showerror("Error", "Failed to add file to dataset")
                self.status_var.set("Failed to add file to dataset")
        except Exception as e:
            messagebox.showerror("Error", f"Error submitting feedback: {str(e)}")
            self.status_var.set("Error submitting feedback")

    def _evaluate_accuracy(self):
        """Evaluate the accuracy of the raga identification system."""
        # Evaluate accuracy
        metrics = self.dataset_manager.evaluate_accuracy()
        
        # Enable text widget for editing
        self.metrics_text.config(state=tk.NORMAL)
        
        # Clear previous content
        self.metrics_text.delete(1.0, tk.END)
        
        # Add metrics
        if metrics and metrics.get('total', 0) > 0:
            self.metrics_text.insert(tk.END, f"Total files: {metrics['total']}\n")
            self.metrics_text.insert(tk.END, f"Correct: {metrics['correct']}\n")
            self.metrics_text.insert(tk.END, f"Accuracy: {metrics['accuracy']:.2%}\n")
            self.metrics_text.insert(tk.END, f"Top-3 accuracy: {metrics['top_3_accuracy']:.2%}\n")
        else:
            self.metrics_text.insert(tk.END, "No metrics available or no files in dataset")
        
        # Make read-only again
        self.metrics_text.config(state=tk.DISABLED)
        
        # Update status
        self.status_var.set("Accuracy evaluation complete")

    def _generate_report(self):
        """Generate a report of the dataset and model performance."""
        # Generate report
        report_path = self.dataset_manager.generate_report()
        
        if report_path:
            messagebox.showinfo("Report Generated", f"Report saved to: {report_path}")
            self.status_var.set(f"Report saved to: {report_path}")
            
            # Try to open the report
            try:
                if sys.platform == 'darwin':  # macOS
                    subprocess.run(['open', report_path])
                elif sys.platform == 'win32':  # Windows
                    os.startfile(report_path)
                else:  # Linux
                    subprocess.run(['xdg-open', report_path])
            except:
                pass
        else:
            messagebox.showerror("Error", "Failed to generate report")
            self.status_var.set("Failed to generate report")
    def _update_output_display(self, raga_name, generated_files):
        """Update the output display with generated files."""
        # Enable text widget for editing
        self.output_text.config(state=tk.NORMAL)
        
        # Clear previous content
        self.output_text.delete(1.0, tk.END)
        
        # Add generation info
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.output_text.insert(tk.END, f"Generated {raga_name} Lo-Fi components at {timestamp}\n\n", "title")
        
        # Add file paths
        for component, filepath in generated_files.items():
            self.output_text.insert(tk.END, f"{component}:\n", "title")
            self.output_text.insert(tk.END, f"{filepath}\n\n", "path")
        
        # Make read-only again
        self.output_text.config(state=tk.DISABLED)
    
    def _open_output_folder(self):
        """Open the output folder in file explorer."""
        output_dir = "outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Open folder based on platform
        if sys.platform == 'darwin':  # macOS
            subprocess.run(['open', output_dir])
        elif sys.platform == 'win32':  # Windows
            os.startfile(output_dir)
        else:  # Linux
            subprocess.run(['xdg-open', output_dir])
    
    def _open_in_logic(self):
        """Import the generated files to Logic Pro X."""
        # Only available on macOS
        if sys.platform != 'darwin':
            messagebox.showerror("Error", "This feature is only available on macOS")
            return
            
        # Check if there are generated files
        self.output_text.config(state=tk.NORMAL)
        text_content = self.output_text.get(1.0, tk.END)
        self.output_text.config(state=tk.DISABLED)
        
        if ".mid" not in text_content:
            messagebox.showinfo("No Files", "Please generate some MIDI files first")
            return
        
        # Try to open Logic Pro X with the files
        try:
            # Find all MIDI files in the output
            import re
            paths = re.findall(r'(outputs/.*\.mid)', text_content)
            
            if not paths:
                messagebox.showinfo("No Files", "Could not find valid MIDI file paths")
                return
                
            # Convert to absolute paths
            abs_paths = [os.path.abspath(p) for p in paths]
            
            # Open Logic Pro X with the files
            subprocess.run(['open', '-a', 'Logic Pro X'] + abs_paths)
            
            self.status_var.set("Opened files in Logic Pro X")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open Logic Pro X: {str(e)}")
    
    def _save_settings(self):
        """Save current settings to a preset file."""
        # Get current settings
        idx = self.raga_dropdown.current()
        if idx < 0:
            messagebox.showerror("Error", "Please select a raga")
            return
            
        selected_id = self.raga_ids[idx]
        
        settings = {
            'raga_id': selected_id,
            'bpm': self.bpm_var.get(),
            'use_patterns': self.use_patterns_var.get(),
            'base_note': self.base_note_var.get(),
            'octave': self.octave_var.get(),
            'melody_length': self.melody_length_var.get(),
            'generate_melody': self.melody_var.get(),
            'generate_chords': self.chords_var.get(),
            'generate_bass': self.bass_var.get()
        }
        
        # Create presets directory if it doesn't exist
        presets_dir = "presets"
        if not os.path.exists(presets_dir):
            os.makedirs(presets_dir)
            
        # Get filename from user
        filename = filedialog.asksaveasfilename(
            initialdir=presets_dir,
            title="Save Settings",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
            defaultextension=".json"
        )
        
        if not filename:
            return
            
        # Save settings to file
        try:
            with open(filename, 'w') as f:
                json.dump(settings, f, indent=4)
                
            self.status_var.set(f"Settings saved to {os.path.basename(filename)}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save settings: {str(e)}")


if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs("outputs", exist_ok=True)
    
    # Create and run application
    root = tk.Tk()
    app = RagaLoFiApp(root)
    root.mainloop()




