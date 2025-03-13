#!/usr/bin/env python3
"""
Sample Library Setup Script
--------------------------
Utility script to set up the sample library structure and download some free samples.
"""

import os
import sys
import zipfile
import json
import argparse
from urllib.request import urlretrieve
import shutil

def create_sample_library(base_path="sample_library"):
    """Create the directory structure for the sample library."""
    
    # Define the structure
    library_structure = {
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
    
    # Create directories
    for category in library_structure:
        for subcategory in library_structure[category]:
            path = os.path.join(base_path, category, subcategory)
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")
    
    # Create a README file
    readme_path = os.path.join(base_path, "README.md")
    with open(readme_path, 'w') as f:
        f.write("# Lo-Fi Sample Library\n\n")
        f.write("This directory contains samples organized for lo-fi beat production.\n\n")
        f.write("## Structure\n\n")
        for category in library_structure:
            f.write(f"### {category.capitalize()}\n\n")
            for subcategory in library_structure[category]:
                f.write(f"- {subcategory}: Place your {subcategory} samples here\n")
            f.write("\n")
        f.write("\nAdd your own samples to these directories or use the download_samples.py script to add free samples.\n")
    
    print(f"Created README: {readme_path}")
    return library_structure

def download_free_samples(library_path="sample_library"):
    """
    Download some free lo-fi samples to get started.
    
    Note: This is a placeholder function. In a real implementation,
    you would need to ensure you have the rights to download and use these samples.
    """
    print("This is a placeholder for downloading free samples.")
    print("In a real implementation, you would connect to legitimate free sample sources.")
    print("For now, please add your own samples to the library directories.")
    
    # Here you could add code to download from free sample sites that allow redistribution
    # or from your own hosted collection of free samples

def import_splice_samples(splice_dir, library_path="sample_library"):
    """
    Import samples from a Splice download directory into the library.
    
    Parameters:
    - splice_dir: Path to the directory containing Splice samples
    - library_path: Path to the sample library
    """
    if not os.path.exists(splice_dir):
        print(f"Error: Splice directory not found: {splice_dir}")
        return
    
    # Counter for imported files
    imported_count = 0
    
    # Walk through the Splice directory
    for root, _, files in os.walk(splice_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.aiff')):
                file_path = os.path.join(root, file)
                file_lower = file.lower()
                
                # Determine target directory based on filename or path
                target_dir = None
                
                # Check for drums
                if any(keyword in file_lower for keyword in ['kick', 'bass drum', 'bd']):
                    target_dir = os.path.join(library_path, "drums", "kicks")
                elif any(keyword in file_lower for keyword in ['snare', 'sd', 'clap']):
                    target_dir = os.path.join(library_path, "drums", "snares")
                elif any(keyword in file_lower for keyword in ['hat', 'hh', 'hihat']):
                    target_dir = os.path.join(library_path, "drums", "hats")
                elif any(keyword in file_lower for keyword in ['perc', 'percussion', 'shaker', 'tambourine']):
                    target_dir = os.path.join(library_path, "drums", "percs")
                elif any(keyword in file_lower for keyword in ['loop', 'drum loop']):
                    target_dir = os.path.join(library_path, "drums", "loops")
                
                # Check for effects
                elif any(keyword in file_lower for keyword in ['vinyl', 'crackle', 'noise']):
                    target_dir = os.path.join(library_path, "effects", "vinyl")
                elif any(keyword in file_lower for keyword in ['foley', 'sfx']):
                    target_dir = os.path.join(library_path, "effects", "foley")
                elif any(keyword in file_lower for keyword in ['ambient', 'ambience', 'background']):
                    target_dir = os.path.join(library_path, "effects", "ambience")
                
                # Check for oneshots
                elif any(keyword in file_lower for keyword in ['key', 'piano', 'rhodes', 'chord']):
                    target_dir = os.path.join(library_path, "oneshots", "keys")
                elif any(keyword in file_lower for keyword in ['guitar', 'gtr']):
                    target_dir = os.path.join(library_path, "oneshots", "guitar")
                elif any(keyword in file_lower for keyword in ['bass']):
                    target_dir = os.path.join(library_path, "oneshots", "bass")
                
                # Copy the file if a target directory was determined
                if target_dir:
                    # Ensure target directory exists
                    os.makedirs(target_dir, exist_ok=True)
                    
                    # Copy the file
                    target_path = os.path.join(target_dir, file)
                    shutil.copy2(file_path, target_path)
                    imported_count += 1
                    print(f"Imported: {file} -> {os.path.relpath(target_path, library_path)}")
    
    print(f"\nImported {imported_count} samples from Splice directory.")

def import_from_directory(source_dir, category, subcategory, library_path="sample_library"):
    """
    Import samples from a directory into a specific category and subcategory.
    
    Parameters:
    - source_dir: Path to the source directory
    - category: Target category ('drums', 'effects', 'oneshots')
    - subcategory: Target subcategory within the category
    - library_path: Path to the sample library
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found: {source_dir}")
        return
    
    # Validate category and subcategory
    valid_structure = {
        "drums": ["kicks", "snares", "hats", "percs", "loops"],
        "effects": ["vinyl", "foley", "ambience"],
        "oneshots": ["keys", "guitar", "bass"]
    }
    
    if category not in valid_structure:
        print(f"Error: Invalid category '{category}'. Valid categories: {list(valid_structure.keys())}")
        return
    
    if subcategory not in valid_structure[category]:
        print(f"Error: Invalid subcategory '{subcategory}' for category '{category}'.")
        print(f"Valid subcategories: {valid_structure[category]}")
        return
    
    # Create target directory
    target_dir = os.path.join(library_path, category, subcategory)
    os.makedirs(target_dir, exist_ok=True)
    
    # Counter for imported files
    imported_count = 0
    
    # Copy all audio files
    for file in os.listdir(source_dir):
        if file.endswith(('.wav', '.mp3', '.aiff', '.ogg')):
            source_path = os.path.join(source_dir, file)
            target_path = os.path.join(target_dir, file)
            
            shutil.copy2(source_path, target_path)
            imported_count += 1
            print(f"Imported: {file} -> {category}/{subcategory}")
    
    print(f"\nImported {imported_count} samples to {category}/{subcategory}.")

def main():
    parser = argparse.ArgumentParser(description='Set up the Lo-Fi sample library.')
    
    # Main command
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Create library command
    create_parser = subparsers.add_parser('create', help='Create the sample library structure')
    create_parser.add_argument('--path', default='sample_library', help='Path for the sample library')
    
    # Import from Splice command
    splice_parser = subparsers.add_parser('import-splice', help='Import samples from a Splice directory')
    splice_parser.add_argument('splice_dir', help='Path to the Splice download directory')
    splice_parser.add_argument('--path', default='sample_library', help='Path to the sample library')
    
    # Import from directory command
    import_parser = subparsers.add_parser('import-dir', help='Import samples from a directory')
    import_parser.add_argument('source_dir', help='Source directory with samples')
    import_parser.add_argument('category', help='Target category (drums, effects, oneshots)')
    import_parser.add_argument('subcategory', help='Target subcategory')
    import_parser.add_argument('--path', default='sample_library', help='Path to the sample library')
    
    args = parser.parse_args()
    
    # Execute the requested command
    if args.command == 'create':
        create_sample_library(args.path)
        print(f"Sample library structure created at: {args.path}")
        
    elif args.command == 'import-splice':
        import_splice_samples(args.splice_dir, args.path)
        
    elif args.command == 'import-dir':
        import_from_directory(args.source_dir, args.category, args.subcategory, args.path)
        
    else:
        # Default to creating the library if no command specified
        create_sample_library()
        print("Sample library structure created at: sample_library")

if __name__ == "__main__":
    main()