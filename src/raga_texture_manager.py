#!/usr/bin/env python3
"""
Raga-Specific Texture Manager
----------------------------
Utility to organize and manage ambient textures for specific ragas.
"""

import os
import sys
import json
import argparse
import shutil

def list_ragas_and_textures(textures_path="sample_library/textures"):
    """List all ragas and their associated texture categories."""
    # Load the mapping file
    mapping_file = os.path.join(textures_path, "raga_mood_mappings.json")
    
    if not os.path.exists(mapping_file):
        print(f"Error: Mapping file not found: {mapping_file}")
        return
    
    try:
        with open(mapping_file, 'r') as f:
            mappings = json.load(f)
    except json.JSONDecodeError:
        print("Error reading mapping file.")
        return
    
    # Count textures in each category
    texture_counts = {}
    category_dirs = [d for d in os.listdir(textures_path) 
                   if os.path.isdir(os.path.join(textures_path, d))]
    
    for category in category_dirs:
        category_path = os.path.join(textures_path, category)
        texture_counts[category] = len([f for f in os.listdir(category_path) 
                                       if f.endswith(('.wav', '.mp3', '.aiff', '.ogg'))])
    
    # Group ragas by mood
    ragas_by_mood = {}
    for raga_id, moods in mappings.items():
        for mood in moods:
            if mood not in ragas_by_mood:
                ragas_by_mood[mood] = []
            ragas_by_mood[mood].append(raga_id)
    
    # Print results
    print("\nRagas by Texture Category:")
    print("==========================")
    
    for mood, ragas in sorted(ragas_by_mood.items()):
        texture_count = texture_counts.get(mood, 0)
        print(f"\n{mood.capitalize()} ({texture_count} textures):")
        print("-" * (len(mood) + 12 + len(str(texture_count))))
        
        for raga_id in sorted(ragas):
            print(f"- {raga_id}")
    
    print("\nTexture Counts by Category:")
    print("==========================")
    
    for category, count in sorted(texture_counts.items()):
        print(f"{category.capitalize()}: {count} textures")

def add_raga_mapping(raga_id, categories, textures_path="sample_library/textures"):
    """
    Add or update a raga mapping to specific texture categories.
    
    Parameters:
    - raga_id: ID of the raga
    - categories: List of texture categories to associate with the raga
    - textures_path: Path to the textures directory
    """
    # Load the mapping file
    mapping_file = os.path.join(textures_path, "raga_mood_mappings.json")
    
    if not os.path.exists(mapping_file):
        print(f"Error: Mapping file not found: {mapping_file}")
        return
    
    try:
        with open(mapping_file, 'r') as f:
            mappings = json.load(f)
    except json.JSONDecodeError:
        print("Error reading mapping file.")
        return
    
    # Validate categories
    valid_categories = [d for d in os.listdir(textures_path) 
                      if os.path.isdir(os.path.join(textures_path, d))]
    
    invalid_categories = [c for c in categories if c not in valid_categories]
    if invalid_categories:
        print(f"Warning: Invalid categories: {', '.join(invalid_categories)}")
        print(f"Valid categories: {', '.join(valid_categories)}")
        
    valid_input_categories = [c for c in categories if c in valid_categories]
    
    # Update the mapping
    mappings[raga_id] = valid_input_categories
    
    # Save the mapping file
    with open(mapping_file, 'w') as f:
        json.dump(mappings, f, indent=2, sort_keys=True)
    
    print(f"Updated mapping for raga '{raga_id}': {', '.join(valid_input_categories)}")

def recommend_textures(raga_id, textures_path="sample_library/textures"):
    """Recommend textures for a specific raga."""
    # Load the mapping file
    mapping_file = os.path.join(textures_path, "raga_mood_mappings.json")
    
    if not os.path.exists(mapping_file):
        print(f"Error: Mapping file not found: {mapping_file}")
        return
    
    try:
        with open(mapping_file, 'r') as f:
            mappings = json.load(f)
    except json.JSONDecodeError:
        print("Error reading mapping file.")
        return
    
    # Check if raga is in mappings
    if raga_id not in mappings:
        print(f"Raga '{raga_id}' not found in mappings.")
        return
    
    # Get the texture categories for this raga
    categories = mappings[raga_id]
    
    print(f"\nRecommended Textures for Raga '{raga_id}':")
    print("=" * (32 + len(raga_id)))
    
    for category in categories:
        category_path = os.path.join(textures_path, category)
        
        if not os.path.exists(category_path):
            continue
            
        textures = [f for f in os.listdir(category_path) 
                   if f.endswith(('.wav', '.mp3', '.aiff', '.ogg'))]
        
        if textures:
            print(f"\n{category.capitalize()} textures ({len(textures)}):")
            print("-" * (len(category) + 11 + len(str(len(textures)))))
            
            for texture in textures:
                print(f"- {texture}")
        else:
            print(f"\n{category.capitalize()} textures: None available")

def copy_to_working_directory(raga_id, target_dir, textures_path="sample_library/textures"):
    """
    Copy recommended textures for a raga to a working directory.
    
    Parameters:
    - raga_id: ID of the raga
    - target_dir: Target directory to copy textures to
    - textures_path: Path to the textures directory
    """
    # Load the mapping file
    mapping_file = os.path.join(textures_path, "raga_mood_mappings.json")
    
    if not os.path.exists(mapping_file):
        print(f"Error: Mapping file not found: {mapping_file}")
        return
    
    try:
        with open(mapping_file, 'r') as f:
            mappings = json.load(f)
    except json.JSONDecodeError:
        print("Error reading mapping file.")
        return
    
    # Check if raga is in mappings
    if raga_id not in mappings:
        print(f"Raga '{raga_id}' not found in mappings.")
        return
    
    # Get the texture categories for this raga
    categories = mappings[raga_id]
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy textures
    copied = 0
    
    for category in categories:
        category_path = os.path.join(textures_path, category)
        
        if not os.path.exists(category_path):
            continue
            
        textures = [f for f in os.listdir(category_path) 
                   if f.endswith(('.wav', '.mp3', '.aiff', '.ogg'))]
        
        for texture in textures:
            source_path = os.path.join(category_path, texture)
            target_path = os.path.join(target_dir, f"{category}_{texture}")
            
            shutil.copy2(source_path, target_path)
            copied += 1
            print(f"Copied: {texture} -> {target_path}")
    
    print(f"\nCopied {copied} textures for raga '{raga_id}' to {target_dir}")

def main():
    parser = argparse.ArgumentParser(description='Manage ambient textures for ragas.')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List ragas and textures')
    list_parser.add_argument('--textures-path', default='sample_library/textures', 
                          help='Path to textures directory')
    
    # Add mapping command
    add_parser = subparsers.add_parser('map', help='Add or update a raga mapping')
    add_parser.add_argument('raga_id', help='ID of the raga')
    add_parser.add_argument('categories', nargs='+', help='Texture categories to associate with the raga')
    add_parser.add_argument('--textures-path', default='sample_library/textures', 
                         help='Path to textures directory')
    
    # Recommend command
    recommend_parser = subparsers.add_parser('recommend', help='Recommend textures for a raga')
    recommend_parser.add_argument('raga_id', help='ID of the raga')
    recommend_parser.add_argument('--textures-path', default='sample_library/textures', 
                               help='Path to textures directory')
    
    # Copy command
    copy_parser = subparsers.add_parser('copy', help='Copy recommended textures to working directory')
    copy_parser.add_argument('raga_id', help='ID of the raga')
    copy_parser.add_argument('target_dir', help='Target directory to copy textures to')
    copy_parser.add_argument('--textures-path', default='sample_library/textures', 
                          help='Path to textures directory')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_ragas_and_textures(args.textures_path)
    
    elif args.command == 'map':
        add_raga_mapping(args.raga_id, args.categories, args.textures_path)
    
    elif args.command == 'recommend':
        recommend_textures(args.raga_id, args.textures_path)
    
    elif args.command == 'copy':
        copy_to_working_directory(args.raga_id, args.target_dir, args.textures_path)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()