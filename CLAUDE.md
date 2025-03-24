# CLAUDE.md: Raga Lo-Fi Toolkit

## Setup
- Python 3.6+ environment required
- Install dependencies: `pip install numpy librosa pydub soundfile mido matplotlib scipy tqdm`
- For GUI: `pip install tkinter`
- For AudioCraft: Requires PyTorch and TorchAudio

## Code Style
- Follow PEP 8 with 4 spaces, 100 char line length
- Classes: CamelCase (e.g., `LofiStackGenerator`)
- Functions/methods: snake_case (e.g., `generate_from_raga`)
- Private methods: prefix with underscore (e.g., `_load_sample_library`)
- Constants: UPPER_CASE

## Import Conventions
- Standard library → third-party → local modules
- Group imports by category with blank line between
- Use explicit imports (avoid `from x import *`)

## Error Handling
- Use try/except blocks with descriptive messages
- Return None for failed operations, not empty structures

## Documentation
- Google-style docstrings for all public methods
- Include type hints for parameters
- Add examples for complex functionality

## Testing
- Run tests with: `python -m unittest discover tests`
- Single test: `python -m unittest tests.test_module.TestCase`

## Completed Module Implementation Requests

### Batch 1: Harmony Analyzer ✅
- Implement `harmony_analyzer.py` for analyzing chord structures in audio/MIDI
- Feature: Chord detection and progression analysis
- Feature: Mapping raga notes to Western harmony
- Feature: Generate compatible chord progressions for ragas

### Batch 2: AudioCraft Bridge ✅
- Implement `audiocraft_bridge.py` for safe integration with MusicGen
- Feature: Dynamic path management for external AudioCraft installation
- Feature: Error handling for missing dependencies with graceful fallbacks
- Feature: Raga-informed prompt construction from harmony analysis
- Feature: Support for reference audio conditioning
- Feature: Batch processing for generating variations

Example usage:
```python
from audiocraft_bridge import get_music_gen
from harmony_analyzer import HarmonyAnalyzer

# Analyze melody from raga generator
harmony = HarmonyAnalyzer().analyze_melody(midi_file="yaman_melody.mid")

# Get AudioCraft model with parameters derived from analysis
model = get_music_gen(model_size='small')

# Generate with harmony-informed parameters
audio = model.generate_from_analysis(
    harmony_analysis=harmony,
    duration=30,
    variations=2,
    enhancement_level=0.7
)

# Save results
model.save_audio(audio[0], "yaman_backing.wav")
```

## Completed Module Implementation Requests

### Batch 3: Audio Processor ✅
- Implement `audio_processor.py` for audio processing, mixing and applying lo-fi effects
- Feature: Multiple lo-fi style presets (classic, tape, vinyl, ambient)
- Feature: Chord-aware harmonic processing based on harmony analysis
- Feature: Comprehensive effect chain (filtering, saturation, vinyl noise, reverb)
- Feature: Integration with AudioCraft bridge for AI-enhanced audio generation
- Feature: Audio normalization, file saving, and effects chain export

## Completed Module Implementation Requests

### Batch 4: Main Application ✅
- Implement `main.py` to orchestrate the complete workflow
- Feature: Command-line argument parsing and configuration management
- Feature: Pipeline from raga selection to final audio export
- Feature: Complete workflow for generating raga-based lo-fi music
- Feature: Error handling and reporting

Example usage:
```bash
# List available ragas
python src/main.py list-ragas

# Generate a track from a raga
python src/main.py generate --raga yaman --bpm 75 --key C

# Generate with AI integration
python src/main.py generate --raga darbari --use-ai

# Process existing audio
python src/main.py process path/to/audio.wav --style vinyl

# Generate with AudioCraft
python src/main.py audiocraft "peaceful lo-fi beats with piano"
```

