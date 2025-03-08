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
from enhanced_raga_generator import EnhancedRagaGenerator

class RagaLoFiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Raga Lo-Fi Generator")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)
        
        # Initialize the generator
        self.generator = EnhancedRagaGenerator()
        
        # Set up main frame
        self.main_frame = ttk.Frame(root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create UI elements
        self._create_header()
        self._create_raga_selection()
        self._create_parameters()
        self._create_generation_controls()
        self._create_output_display()
        self._create_status_bar()
        
        # Set default values
        self._set_defaults()
        
        # Configure styling
        self._configure_styles()
    
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
        """Create the raga selection section."""
        raga_frame = ttk.LabelFrame(self.main_frame, text="Raga Selection", padding=10)
        raga_frame.pack(fill=tk.X, pady=(0, 15))
        
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
        self.raga_options = [f"{raga['name']} - {raga['mood']}" for raga in self.available_ragas]
        self.raga_ids = [raga['id'] for raga in self.available_ragas]
        
        self.selected_raga = tk.StringVar()
        self.raga_dropdown = ttk.Combobox(
            direct_frame, 
            textvariable=self.selected_raga,
            values=self.raga_options,
            width=50,
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
            width=50,
            state="readonly"
        )
        self.time_ragas_dropdown.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        self.time_ragas_dropdown.bind("<<ComboboxSelected>>", self._on_time_raga_selected)
        
        # Tab 3: Selection by mood
        mood_frame = ttk.Frame(self.selection_tabs, padding=10)
        self.selection_tabs.add(mood_frame, text="By Mood")
        
        # Mood selection
        ttk.Label(mood_frame, text="Mood:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.mood_options = ["focus", "deep_concentration", "relaxation", "creative", "meditative"]
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
            width=50,
            state="readonly"
        )
        self.mood_ragas_dropdown.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        self.mood_ragas_dropdown.bind("<<ComboboxSelected>>", self._on_mood_raga_selected)
    
    def _create_parameters(self):
        """Create parameter controls."""
        params_frame = ttk.LabelFrame(self.main_frame, text="Generation Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Create two columns
        left_frame = ttk.Frame(params_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_frame = ttk.Frame(params_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
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
        self.use_patterns_var = tk.BooleanVar()
        ttk.Checkbutton(
            left_frame, 
            text="Use characteristic patterns", 
            variable=self.use_patterns_var
        ).pack(anchor=tk.W, pady=5)
        
        # Right column - Base note
        base_note_frame = ttk.Frame(right_frame)
        base_note_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(base_note_frame, text="Base note:").pack(side=tk.LEFT)
        
        self.base_notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        self.base_note_var = tk.StringVar()
        self.base_note_dropdown = ttk.Combobox(
            base_note_frame, 
            textvariable=self.base_note_var,
            values=self.base_notes,
            width=5,
            state="readonly"
        )
        self.base_note_dropdown.pack(side=tk.LEFT, padx=5)
        
        self.octave_var = tk.IntVar()
        self.octave_dropdown = ttk.Combobox(
            base_note_frame, 
            textvariable=self.octave_var,
            values=[3, 4, 5],
            width=3,
            state="readonly"
        )
        self.octave_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Right column - Melody length
        melody_length_frame = ttk.Frame(right_frame)
        melody_length_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(melody_length_frame, text="Melody length:").pack(side=tk.LEFT)
        
        self.melody_length_var = tk.IntVar()
        self.melody_length_dropdown = ttk.Combobox(
            melody_length_frame, 
            textvariable=self.melody_length_var,
            values=[16, 24, 32, 48, 64],
            width=5,
            state="readonly"
        )
        self.melody_length_dropdown.pack(side=tk.LEFT, padx=5)
        ttk.Label(melody_length_frame, text="notes").pack(side=tk.LEFT)
    
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
    
    def _on_raga_selected(self, event):
        """Update display when raga is selected."""
        if not self.selected_raga.get():
            return
            
        # Find the selected raga
        idx = self.raga_dropdown.current()
        if idx >= 0:
            selected_id = self.raga_ids[idx]
            raga_data = self.generator.ragas_data.get(selected_id, {})
            
            # Update description
            description = f"Time: {raga_data.get('time', 'N/A')}\n"
            description += f"Mood: {raga_data.get('mood', 'N/A')}\n"
            description += f"Suitable for: {', '.join(raga_data.get('suitable_for', ['N/A']))}"
            
            self.raga_description.config(text=description)
    
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
    
    def _generate_thread(self, raga_id, base_note, bpm, use_patterns, melody_length):
        """Thread function for file generation."""
        try:
            generated_files = {}
            
            # Generate selected components
            if self.melody_var.get():
                melody_file = self.generator.generate_melody(
                    raga_id, 
                    length=melody_length,
                    use_patterns=use_patterns,
                    base_note=base_note,
                    bpm=bpm
                )
                generated_files['Melody'] = melody_file
            
            if self.chords_var.get():
                chord_file = self.generator.generate_chord_progression(
                    raga_id,
                    length=4,
                    base_note=base_note-12,
                    bpm=bpm
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