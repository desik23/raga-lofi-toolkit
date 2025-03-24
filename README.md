# Raga Lo-Fi Toolkit

A comprehensive toolkit for generating lo-fi music based on Indian ragas, bridging traditional music theory with modern production techniques.

## Features

- Generate authentic raga-based melodies, chord progressions, and bass lines
- Analyze and identify raga characteristics in existing audio
- Process audio with various lo-fi effect styles (classic, tape, vinyl, ambient)
- Generate AI-enhanced audio using AudioCraft's MusicGen
- Complete workflow from raga selection to final lo-fi production

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/raga-lofi-toolkit.git
cd raga-lofi-toolkit
```

2. Install the required dependencies:
```bash
pip install numpy librosa pydub soundfile mido matplotlib scipy tqdm
```

3. Optional: Install AudioCraft for AI audio generation:
```bash
git clone https://github.com/facebookresearch/audiocraft.git ~/audiocraft
cd ~/audiocraft
pip install -e .
```

## Usage

### Command Line Interface

The toolkit provides a comprehensive CLI for generating and processing raga-based lo-fi music:

#### List Available Ragas
```bash
python src/main.py list-ragas
```

#### Generate a Lo-Fi Track from a Raga
```bash
python src/main.py generate --raga yaman --bpm 75 --key C
```

#### Generate with AudioCraft Integration
```bash
python src/main.py generate --raga darbari --use-ai
```

#### Process Existing Audio with Lo-Fi Effects
```bash
python src/main.py process path/to/audio.wav --style vinyl --intensity 0.6
```

#### Generate Audio with AudioCraft
```bash
python src/main.py audiocraft "peaceful lo-fi beats with piano and soft drums"
```

### Configuration

You can customize the toolkit by editing the `config.json` file, which allows you to:
- Configure AudioCraft settings (model size, device)
- Set default audio processing parameters
- Adjust generation defaults (BPM, key, effect styles)
- Set custom paths for outputs and sample libraries

## Core Components

- **EnhancedRagaGenerator**: Creates authentic raga-based melodies and patterns
- **HarmonyAnalyzer**: Analyzes chord structures and maps ragas to Western harmony
- **AudioCraftBridge**: Integrates with MusicGen for AI audio generation
- **AudioProcessor**: Applies lo-fi effects and transforms audio
- **LofiStackGenerator**: Orchestrates complete track generation

## Example Workflow

1. Select a raga (e.g., Yaman, Bhairav)
2. Generate melody, chord progression, and bass line
3. Analyze harmony structure
4. Apply lo-fi effects
5. Optionally enhance with AI-generated elements
6. Export as complete lo-fi track

## License

[MIT License](LICENSE)

## Acknowledgments

- Indian classical music traditions (Hindustani and Carnatic)
- AudioCraft project by Meta Research
- Lo-fi hip-hop community for inspiration