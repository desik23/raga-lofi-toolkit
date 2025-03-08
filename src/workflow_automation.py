#!/usr/bin/env python3
# workflow_automation.py

import os
import subprocess
import json
import sys
from datetime import datetime
from raga_generator import RagaMelodyGenerator

def main():
    # Load ragas data
    with open('data/ragas.json', 'r') as f:
        data = json.load(f)
    
    # List available ragas
    print("Available ragas:")
    for i, raga in enumerate(data['ragas']):
        print(f"{i+1}. {raga['name']} - {raga['mood']} ({raga['time']})")
    
    # Get user selection
    selection = int(input("\nSelect a raga (number): ")) - 1
    if selection < 0 or selection >= len(data['ragas']):
        print("Invalid selection")
        return
    
    selected_raga = data['ragas'][selection]
    
    # Create output directory with date
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_dir = f"outputs/{date_str}_{selected_raga['name'].lower()}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate content
    generator = RagaMelodyGenerator()
    
    # Generate multiple variations
    variations = 3
    for i in range(variations):
        # Generate melody
        melody_file = generator.generate_melody(
            selected_raga['name'].lower(), 
            length=32,
            bpm=75
        )
        
        # Generate chord progression
        chord_file = generator.generate_chord_progression(
            selected_raga['name'].lower(),
            length=4,
            bpm=75
        )
        
        # Move files to output directory
        os.rename(melody_file, f"{output_dir}/variation_{i+1}_{melody_file}")
        os.rename(chord_file, f"{output_dir}/variation_{i+1}_{chord_file}")
    
    print(f"\nGenerated {variations} variations in {output_dir}")
    
    # Try to open the output folder
    try:
        if sys.platform == 'darwin':  # macOS
            subprocess.run(['open', output_dir])
        elif sys.platform == 'win32':  # Windows
            subprocess.run(['explorer', output_dir.replace('/', '\\')])
        else:  # Linux
            subprocess.run(['xdg-open', output_dir])
    except:
        pass

if __name__ == "__main__":
    main()