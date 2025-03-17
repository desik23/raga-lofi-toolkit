#!/usr/bin/env python3
"""
Raga Lo-Fi Beat Generator GUI
----------------------------
Simple GUI for generating lo-fi beats based on Indian ragas.
"""

import os
import sys
import json
import time
import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

# Import the generator
from lofi_stack_generator import LofiStackGenerator

class LofiGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Raga Lo-Fi Beat Generator")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)
        
        # Initialize the generator
        self.generator = LofiStackGenerator()
        
        # Create the UI
        self._create_ui()
    
    def _create_ui(self):
        """Create the user interface."""
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(
            header_frame, 
            text="Raga Lo-Fi Beat Generator", 
            font=("Helvetica", 16, "bold")
        ).pack(side=tk.LEFT)
        
        # Create notebook for different generation modes
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Tab 1: Generate from Raga
        self.raga_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.raga_frame, text="From Raga")
        self._create_raga_tab()
        
        # Tab 2: Generate from Melody
        self.melody_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.melody_frame, text="From Melody")
        self._create_melody_tab()
        
        # Tab 3: Generate from Loop
        self.loop_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.loop_frame, text="From Loop")
        self._create_loop_tab()
        
        # Common settings panel
        settings_frame = ttk.LabelFrame(main_frame, text="Common Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 20))
        self._create_settings_panel(settings_frame)
        
        # Output display
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True)
        self._create_output_display(output_frame)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _create_raga_tab(self):
        """Create the 'Generate from Raga' tab."""
        # Raga selection frame
        raga_select_frame = ttk.Frame(self.raga_frame)
        raga_select_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(raga_select_frame, text="Select Raga:").pack(side=tk.LEFT, padx=(0, 10))
        
        # Get available ragas
        available_ragas = self.generator.raga_generator.list_available_ragas()
        self.raga_options = []
        self.raga_ids = []
        
        for raga in available_ragas:
            self.raga_options.append(f"{raga['name']} - {raga['mood']}")
            self.raga_ids.append(raga['id'])
        
        self.selected_raga = tk.StringVar()
        self.raga_dropdown = ttk.Combobox(
            raga_select_frame, 
            textvariable=self.selected_raga,
            values=self.raga_options,
            width=40,
            state="readonly"
        )
        self.raga_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Random raga button
        ttk.Button(
            raga_select_frame,
            text="Random Raga",
            command=self._select_random_raga
        ).pack(side=tk.LEFT, padx=5)
        
        # Raga description
        self.raga_description = ttk.Label(self.raga_frame, text="", wraplength=700)
        self.raga_description.pack(fill=tk.X, pady=(0, 20))
        self.raga_dropdown.bind("<<ComboboxSelected>>", self._on_raga_selected)
        
        # Generation button
        generate_button = ttk.Button(
            self.raga_frame,
            text="Generate From Raga",
            command=self._generate_from_raga,
            style="Generate.TButton"
        )
        generate_button.pack(fill=tk.X)
    
    def _create_melody_tab(self):
        """Create the 'Generate from Melody' tab."""
        # Melody file selection
        melody_frame = ttk.Frame(self.melody_frame)
        melody_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(melody_frame, text="Melody File:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.melody_path_var = tk.StringVar()
        melody_entry = ttk.Entry(melody_frame, textvariable=self.melody_path_var, width=40)
        melody_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        browse_button = ttk.Button(
            melody_frame,
            text="Browse...",
            command=self._browse_melody
        )
        browse_button.pack(side=tk.LEFT)
        
        # Options
        options_frame = ttk.Frame(self.melody_frame)
        options_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.analyze_melody_var = tk.BooleanVar(value=True)
        analyze_check = ttk.Checkbutton(
            options_frame,
            text="Attempt to identify raga in melody",
            variable=self.analyze_melody_var
        )
        analyze_check.pack(anchor=tk.W)
        
        # Generation button
        generate_button = ttk.Button(
            self.melody_frame,
            text="Generate From Melody",
            command=self._generate_from_melody,
            style="Generate.TButton"
        )
        generate_button.pack(fill=tk.X)
    
    def _create_loop_tab(self):
        """Create the 'Generate from Loop' tab."""
        # Loop file selection
        loop_frame = ttk.Frame(self.loop_frame)
        loop_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(loop_frame, text="Loop File:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.loop_path_var = tk.StringVar()
        loop_entry = ttk.Entry(loop_frame, textvariable=self.loop_path_var, width=40)
        loop_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        browse_button = ttk.Button(
            loop_frame,
            text="Browse...",
            command=self._browse_loop
        )
        browse_button.pack(side=tk.LEFT)
        
        # What to generate checkbox grid
        generate_frame = ttk.LabelFrame(self.loop_frame, text="Generate Components")
        generate_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.loop_generate_melody_var = tk.BooleanVar(value=True)
        melody_check = ttk.Checkbutton(
            generate_frame,
            text="Generate Melody",
            variable=self.loop_generate_melody_var
        )
        melody_check.grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        
        self.loop_generate_chords_var = tk.BooleanVar(value=True)
        chords_check = ttk.Checkbutton(
            generate_frame,
            text="Generate Chords",
            variable=self.loop_generate_chords_var
        )
        chords_check.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        self.loop_generate_bass_var = tk.BooleanVar(value=True)
        bass_check = ttk.Checkbutton(
            generate_frame,
            text="Generate Bass",
            variable=self.loop_generate_bass_var
        )
        bass_check.grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        
        self.loop_generate_drums_var = tk.BooleanVar(value=True)
        drums_check = ttk.Checkbutton(
            generate_frame,
            text="Generate Drums",
            variable=self.loop_generate_drums_var
        )
        drums_check.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Generation button
        generate_button = ttk.Button(
            self.loop_frame,
            text="Generate From Loop",
            command=self._generate_from_loop,
            style="Generate.TButton"
        )
        generate_button.pack(fill=tk.X)
    
    def _create_settings_panel(self, parent_frame):
        """Create common settings panel."""
        # Create a grid
        settings_grid = ttk.Frame(parent_frame)
        settings_grid.pack(fill=tk.X)
        
        # BPM setting
        ttk.Label(settings_grid, text="BPM:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        
        self.bpm_var = tk.IntVar(value=75)
        bpm_spinbox = ttk.Spinbox(
            settings_grid,
            from_=60,
            to=90,
            textvariable=self.bpm_var,
            width=5
        )
        bpm_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Key setting
        ttk.Label(settings_grid, text="Key:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        self.key_var = tk.StringVar(value="C")
        key_dropdown = ttk.Combobox(
            settings_grid,
            textvariable=self.key_var,
            values=keys,
            width=5,
            state="readonly"
        )
        key_dropdown.grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Length setting
        ttk.Label(settings_grid, text="Length:").grid(row=0, column=4, sticky=tk.W, padx=5, pady=2)
        
        self.length_var = tk.IntVar(value=32)
        length_spinbox = ttk.Spinbox(
            settings_grid,
            from_=16,
            to=64,
            increment=8,
            textvariable=self.length_var,
            width=5
        )
        length_spinbox.grid(row=0, column=5, sticky=tk.W, padx=5, pady=2)
        
        # Output directory
        ttk.Label(settings_grid, text="Output Dir:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        
        self.output_dir_var = tk.StringVar(value="outputs")
        output_entry = ttk.Entry(
            settings_grid,
            textvariable=self.output_dir_var,
            width=30
        )
        output_entry.grid(row=1, column=1, columnspan=3, sticky=tk.W+tk.E, padx=5, pady=2)
        
        browse_button = ttk.Button(
            settings_grid,
            text="Browse...",
            command=self._browse_output_dir
        )
        browse_button.grid(row=1, column=4, sticky=tk.W, padx=5, pady=2)
        
        # Open output folder button
        open_button = ttk.Button(
            settings_grid,
            text="Open Output Folder",
            command=self._open_output_folder
        )
        open_button.grid(row=1, column=5, sticky=tk.W, padx=5, pady=2)
    
    def _create_output_display(self, parent_frame):
        """Create output display area."""
        # Create a Text widget with scrollbar
        self.output_text = tk.Text(parent_frame, wrap=tk.WORD, height=10)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(parent_frame, command=self.output_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.output_text.config(yscrollcommand=scrollbar.set)
        self.output_text.tag_configure("title", font=("Helvetica", 11, "bold"))
        self.output_text.tag_configure("path", font=("Courier", 10))
        
        # Make read-only
        self.output_text.config(state=tk.DISABLED)
    
    def _select_random_raga(self):
        """Select a random raga from the dropdown."""
        if not self.raga_options:
            return
        
        idx = random.randint(0, len(self.raga_options) - 1)
        self.raga_dropdown.current(idx)
        self._on_raga_selected(None)
    
    def _on_raga_selected(self, event):
        """Update display when raga is selected."""
        if not self.selected_raga.get():
            return
            
        idx = self.raga_dropdown.current()
        if idx >= 0:
            raga_id = self.raga_ids[idx]
            raga_data = self.generator.raga_generator.ragas_data.get(raga_id, {})
            
            description = f"Raga: {raga_data.get('name', raga_id)}\n"
            description += f"System: {raga_data.get('system', 'Hindustani').capitalize()}\n"
            description += f"Mood: {raga_data.get('mood', 'N/A')}\n"
            description += f"Time: {raga_data.get('time', 'N/A')}\n"
            
            if 'suitable_for' in raga_data:
                description += f"Suitable for: {', '.join(raga_data['suitable_for'])}"
            
            self.raga_description.config(text=description)
    
    def _browse_melody(self):
        """Browse for a melody file."""
        file_path = filedialog.askopenfilename(
            title="Select Melody File",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.ogg"),
                ("MIDI Files", "*.mid *.midi"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            self.melody_path_var.set(file_path)
    
    def _browse_loop(self):
        """Browse for a loop file."""
        file_path = filedialog.askopenfilename(
            title="Select Loop File",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.ogg"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            self.loop_path_var.set(file_path)
    
    def _browse_output_dir(self):
        """Browse for an output directory."""
        dir_path = filedialog.askdirectory(title="Select Output Directory")
        
        if dir_path:
            self.output_dir_var.set(dir_path)
    
    def _open_output_folder(self):
        """Open the output folder in the file explorer."""
        output_dir = self.output_dir_var.get()
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Platform-specific folder opening
        if sys.platform == 'win32':
            os.startfile(output_dir)
        elif sys.platform == 'darwin':  # macOS
            import subprocess
            subprocess.Popen(['open', output_dir])
        else:  # Linux
            import subprocess
            subprocess.Popen(['xdg-open', output_dir])
    
    def _generate_from_raga(self):
        """Generate a beat from the selected raga."""
        # Check if a raga is selected
        if not self.selected_raga.get():
            messagebox.showerror("Error", "Please select a raga")
            return
        
        # Get parameters
        idx = self.raga_dropdown.current()
        raga_id = self.raga_ids[idx]
        bpm = self.bpm_var.get()
        key = self.key_var.get()
        length = self.length_var.get()
        output_dir = self.output_dir_var.get()
        
        # Update status
        self.status_var.set(f"Generating beat from raga {raga_id}...")
        
        # Create output dir if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Run in a separate thread to keep UI responsive
        thread = threading.Thread(
            target=self._generate_beat_thread,
            args=("raga", raga_id, bpm, key, length, output_dir)
        )
        thread.daemon = True
        thread.start()
    
    def _generate_from_melody(self):
        """Generate a beat from the selected melody."""
        melody_path = self.melody_path_var.get()
        
        if not melody_path or not os.path.exists(melody_path):
            messagebox.showerror("Error", "Please select a valid melody file")
            return
        
        # Get parameters
        analyze = self.analyze_melody_var.get()
        bpm = self.bpm_var.get()
        key = self.key_var.get()
        output_dir = self.output_dir_var.get()
        
        # Update status
        self.status_var.set(f"Generating beat from melody {os.path.basename(melody_path)}...")
        
        # Create output dir if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Run in a separate thread
        thread = threading.Thread(
            target=self._generate_beat_thread,
            args=("melody", melody_path, bpm, key, 0, output_dir, analyze)
        )
        thread.daemon = True
        thread.start()
    
    def _generate_from_loop(self):
        """Generate a beat from the selected loop."""
        loop_path = self.loop_path_var.get()
        
        if not loop_path or not os.path.exists(loop_path):
            messagebox.showerror("Error", "Please select a valid loop file")
            return
        
        # Get parameters
        bpm = self.bpm_var.get()
        key = self.key_var.get()
        output_dir = self.output_dir_var.get()
        
        # Component options
        components = {
            "melody": self.loop_generate_melody_var.get(),
            "chords": self.loop_generate_chords_var.get(),
            "bass": self.loop_generate_bass_var.get(),
            "drums": self.loop_generate_drums_var.get(),
            "effects": True
        }
        
        # Update status
        self.status_var.set(f"Generating beat from loop {os.path.basename(loop_path)}...")
        
        # Create output dir if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Run in a separate thread
        thread = threading.Thread(
            target=self._generate_beat_thread,
            args=("loop", loop_path, bpm, key, 0, output_dir, True, components)
        )
        thread.daemon = True
        thread.start()
    
    def _generate_beat_thread(self, mode, source, bpm, key, length, output_dir, analyze=True, components=None):
        """
        Thread function for beat generation.
        
        Parameters:
        - mode: "raga", "melody", or "loop"
        - source: raga_id, melody path, or loop path
        - bpm: Beats per minute
        - key: Musical key
        - length: Length of the melody (only for raga mode)
        - output_dir: Output directory
        - analyze: Whether to analyze melody/loop (only for melody/loop mode)
        - components: Dictionary of components to generate (only for loop mode)
        """
        try:
            # Configure generator
            generator = LofiStackGenerator(output_dir=output_dir)
            generator.set_bpm(bpm)
            generator.set_key(key)
            
            # Generate based on mode
            if mode == "raga":
                result = generator.generate_from_raga(
                    raga_id=source,
                    length=length
                )
            elif mode == "melody":
                result = generator.generate_from_melody(
                    melody_file=source,
                    identify_raga=analyze
                )
            elif mode == "loop":
                result = generator.generate_from_loop(
                    loop_file=source,
                    components=components
                )
            else:
                # Should never happen
                self.root.after(0, lambda: self.status_var.set("Error: Invalid generation mode"))
                return
            
            # Update UI with result
            self.root.after(0, lambda: self._update_output_display(result))
            self.root.after(0, lambda: self.status_var.set("Generation complete"))
            
        except Exception as e:
            error_msg = f"Error during generation: {str(e)}"
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            self.root.after(0, lambda: self.status_var.set("Generation failed"))
    
# Add this to the _update_output_display method in the LofiGeneratorApp class

    def _update_output_display(self, result):
        """Update the output display with generation results."""
        if not result:
            return
        
        # Enable text widget for editing
        self.output_text.config(state=tk.NORMAL)
        
        # Clear previous content
        self.output_text.delete(1.0, tk.END)
        
        # Add generation info
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        self.output_text.insert(tk.END, f"Lo-Fi Beat Generated at {timestamp}\n\n", "title")
        
        # Add raga info if available
        if 'raga_id' in result:
            self.output_text.insert(tk.END, f"Raga: {result['raga_name']} ({result['raga_id']})\n")
        elif 'input_melody' in result:
            self.output_text.insert(tk.END, f"Input Melody: {os.path.basename(result['input_melody'])}\n")
            if 'raga_id' in result:
                self.output_text.insert(tk.END, f"Identified Raga: {result['raga_name']} ")
                self.output_text.insert(tk.END, f"(Confidence: {result['raga_confidence']:.2f})\n")
        
        self.output_text.insert(tk.END, f"BPM: {result['bpm']}\n")
        self.output_text.insert(tk.END, f"Key: {result['key']}\n\n")
        
        # Add ambient texture information if available
        if 'texture_moods' in result and result['texture_moods']:
            self.output_text.insert(tk.END, f"Ambient Textures: {', '.join(result['texture_moods'])}\n\n")
        
        # Add file paths
        self.output_text.insert(tk.END, "Generated Files:\n", "title")
        
        for category, files in result['files'].items():
            self.output_text.insert(tk.END, f"{category.capitalize()}:\n")
            
            if isinstance(files, dict):
                for name, path in files.items():
                    # Highlight texture files
                    is_texture = category == "effects" and name == "textures"
                    style = "texture_path" if is_texture else "path"
                    
                    self.output_text.insert(tk.END, f"- {name}: ", "title")
                    
                    if isinstance(path, list):
                        # Handle multiple files (like textures)
                        self.output_text.insert(tk.END, "\n")
                        for i, texture_path in enumerate(path):
                            texture_name = os.path.basename(texture_path)
                            self.output_text.insert(tk.END, f"  {i+1}. {texture_name}\n", style)
                    else:
                        self.output_text.insert(tk.END, f"{path}\n", style)
            elif isinstance(files, str):
                self.output_text.insert(tk.END, f"- {files}\n", "path")
            
            self.output_text.insert(tk.END, "\n")
        
        # Make read-only again
        self.output_text.config(state=tk.DISABLED)
        
        # Show success message
        messagebox.showinfo("Success", "Lo-Fi beat successfully generated!")

        # Add this style configuration in the _create_ui method
        # In the _create_output_display method

        self.output_text.tag_configure("title", font=("Helvetica", 11, "bold"))
        self.output_text.tag_configure("path", font=("Courier", 10))
        self.output_text.tag_configure("texture_path", font=("Courier", 10), foreground="green")

def main():
    # Configure style
    root = tk.Tk()
    style = ttk.Style()
    style.configure("TLabel", font=("Helvetica", 11))
    style.configure("TButton", font=("Helvetica", 11))
    style.configure("Generate.TButton", font=("Helvetica", 12, "bold"), padding=10)
    
    # Create app
    app = LofiGeneratorApp(root)
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()