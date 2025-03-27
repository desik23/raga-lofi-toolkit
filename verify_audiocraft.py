#!/usr/bin/env python3
"""
Utility script to verify AudioCraft installation and configuration.
This helps diagnose issues with external drive configuration.
"""

import os
import sys
import json

def main():
    # Check if the external_drive_config.json exists
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             "external_drive_config.json")
    
    print(f"Checking for external drive configuration at: {config_path}")
    if os.path.exists(config_path):
        print("✅ External drive configuration found")
        
        # Load and display configuration
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print("\nConfiguration contents:")
            print(json.dumps(config, indent=2))
            
            # Check if audiocraft section exists
            if 'audiocraft' in config:
                print("\n✅ AudioCraft configuration found")
                
                # Check model path
                if 'model_path' in config['audiocraft']:
                    model_path = config['audiocraft']['model_path']
                    print(f"Configured model path: {model_path}")
                    
                    # Check if path exists
                    if os.path.exists(model_path):
                        print(f"✅ Model path exists")
                        
                        # Check for audiocraft module
                        audiocraft_init = os.path.join(model_path, "audiocraft", "__init__.py")
                        if os.path.exists(audiocraft_init):
                            print("✅ AudioCraft module found at configured path")
                        else:
                            print("❌ AudioCraft module not found at configured path")
                            print(f"Expected file: {audiocraft_init}")
                    else:
                        print(f"❌ Model path does not exist: {model_path}")
                else:
                    print("❌ No model_path specified in audiocraft configuration")
            else:
                print("❌ No audiocraft section in configuration")
        
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
    else:
        print("❌ External drive configuration not found")
        
        # Create a template configuration
        example_config = {
            "audiocraft": {
                "model_size": "small",
                "device": "cpu",
                "model_path": "/path/to/your/audiocraft"
            },
            "audio_processing": {
                "sample_rate": 44100,
                "bit_depth": 16,
                "channels": 2
            }
        }
        
        print("\nYou can create a file at this location with contents like:")
        print(json.dumps(example_config, indent=2))
    
    # Run the bridge's verification directly
    print("\nRunning full AudioCraft Bridge verification...")
    cmd = f"{sys.executable} src/audiocraft_bridge.py --verify"
    exit_code = os.system(cmd)
    
    if exit_code != 0:
        print("\n❌ Verification failed")
        print("\nTo fix this issue:")
        print("1. Ensure AudioCraft is installed on your system or external drive")
        print("2. Update the model_path in external_drive_config.json to point to your AudioCraft installation")
        print("3. Run this script again to verify")
    else:
        print("\n✅ Verification successful")

if __name__ == "__main__":
    main()