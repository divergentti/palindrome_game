# Palindrome Game (English Version)

**Last Updated: 22 April 2025**

Welcome to the **Palindrome Game**, an interactive application designed to generate, visualize, and play with palindromes in English! This project builds on the insights gained from my previous work on a Finnish palindrome generator ([divergentti/palindrome_generator](https://github.com/divergentti/palindrome_generator)) and extends the concept to English, incorporating a machine learning-inspired approach to generate over 31,000 palindromes using the `initializer.py` script. The game features a PyQt6-based GUI with real-time palindrome validation, scoring, visualization, and feedback to enhance the user experience.

Videos: 
1. [PalindromeGame version 0.0.1](https://youtu.be/5A3nKrbZ9SQ)
2. [PalindromeGame in Windows 11](https://youtu.be/7HxxwZ2sWaU)

## Table of Contents

- Motivation
- Project Overview
- Features
- Installation
- Usage
  - Running the Game
  - Using Pre-Built Binaries
  - Generating New Palindromes
- Development Process
- Generating Palindromes with `initializer.py`
- Challenges and Observations
- Version History
- Future Improvements
- License

## Motivation

The inspiration for this project stems from my previous work on a Finnish palindrome generator ([divergentti/palindrome_generator](https://github.com/divergentti/palindrome_generator)), where I explored the challenges of programmatically generating palindromes in Finnish. Palindromes are fascinating because they require strict symmetry—a word or phrase must read the same forwards and backwards—which poses a unique challenge for computational approaches like machine learning (ML). In the Finnish version, I learned that while ML models (e.g., LSTM) excel at recognizing patterns and probabilities, they struggle with the structural symmetry required for palindromes. This led me to develop a hybrid approach combining logical verification with word suggestion mechanisms, which I adapted and scaled for English in this project.

The English version aims to make palindrome generation and gameplay accessible to a broader audience while improving the user interface and adding interactive features like real-time feedback and visualization.

## Project Overview

The Palindrome Game allows users to input phrases and check if they form palindromes. The application provides real-time feedback, scores the palindromes based on length and "sense" (semantic validity), and visualizes the distribution of palindrome lengths. Over 31,000 palindromes were pre-generated using the `initializer.py` script (located in the `initializer/` directory), which processes English words from various sources (verbs, nouns, adjectives, and books from Project Gutenberg) to create a rich dataset for the game. The resulting `palindromes.json` file is stored in the root directory and used by the game at runtime. The game is built using PyQt6 for the GUI and integrates Matplotlib for visualization.

## Features

- **Interactive GUI**: Built with PyQt6, featuring an input field, LCD displays for length, sense score, and total score, a progress bar, feedback area, and a Matplotlib visualization of palindrome length distribution.
- **Real-Time Palindrome Checking**: As you type, the game highlights matching and non-matching characters (green for matches, red for mismatches, blue for the middle character in odd-length palindromes).
- **Scoring System**: Scores are calculated based on palindrome length and a "sense score" (semantic validity). Bonus points are awarded for known palindromes.
- **Feedback and Suggestions**: Provides detailed feedback on whether the input is a palindrome, suggests extensions, and shows similar palindromes from the pre-generated dataset.
- **LLM-Based Word Suggestions**: Integrates with a large language model (via APIs like LM Studio or Mistral.ai) to suggest words that fit into palindromes, enhancing gameplay by guiding players toward valid and meaningful palindromes.
- **Visualization**: Displays a histogram of palindrome lengths from the pre-generated dataset, with a marker indicating the length of the current input.
- **Pre-Generated Palindromes**: Includes a dataset of over 31,000 palindromes generated using `initializer.py` from English verbs, nouns, adjectives, and book texts, stored in `palindromes.json` in the root directory.
- **Menu Options**: Access game instructions, inspect palindromes, visualize distributions, view player stats, and adjust settings (e.g., number of matching palindromes to show).
- **Player Data Persistence**: Robust saving and loading of player data (total score, playing time, and discovered palindromes) to `players.json`, ensuring stats are preserved across sessions.

## Installation

### Prerequisites

To run the game from source, you’ll need:

- Python 3.8+
- PyQt6
- Matplotlib
- Appdirs
- NLTK (for `initializer.py`)
- Pandas (for `initializer.py`)
- Requests (for LLM integrations and `initializer.py`)

Install the required packages using pip:

```bash
pip install PyQt6 matplotlib nltk pandas requests appdirs
```

**Note**: If you prefer not to set up a Python environment, you can download pre-built binaries for Windows and Linux from the [GitHub Releases page](https://github.com/divergentti/palindrome_game/releases).

### Clone the Repository

```bash
git clone https://github.com/divergentti/palindrome_game.git
cd palindrome_game
```

### Download NLTK Data (for `initializer.py`)

The `initializer.py` script requires NLTK's WordNet data to generate palindromes. Run the following to download it:

```python
import nltk
nltk.download('wordnet')
```

### Configure LLM API (for Word Suggestions)

To enable LLM-based word suggestions, configure the API key for your chosen LLM provider (e.g., LM Studio, Mistral.ai). Update the settings in `settings.json` or provide the API key via environment variables:

```bash
export LLM_API_KEY=<your_api_key>
```

Refer to the provider’s documentation for setup details.

## Usage

### Running the Game

1. Ensure all dependencies are installed and LLM API is configured (if using word suggestions).
2. Run the main game script from the root directory:

   ```bash
   python PalindromeGame.py
   ```
3. The GUI will open, allowing you to:
   - Enter a phrase in the input field to check if it’s a palindrome.
   - View real-time highlighting, scores, and feedback.
   - Use LLM-based word suggestions to extend palindromes (via the feedback area or a dedicated button).
   - Explore the menu options for instructions, stats, and settings.

The game uses `palindromes.json` from the root directory (`./palindromes.json`) and saves player data to `players.json` at runtime.

### Using Pre-Built Binaries

For convenience, pre-built binaries for Windows and Linux are available on the [GitHub Releases page](https://github.com/divergentti/palindrome_game/releases). These binaries were created using GitHub Workflows with Nuitka, a Python compiler that produces standalone executables. The binaries include the `palindromes.json` file, so no additional setup is required.

Note for Windows users: As with all binaries compiled using Nuitka, launching the game may trigger a false positive alert from Windows Defender. The warning typically flags it as Program:Win32/Wacapew.C!ml. This is purely a heuristic detection — there is no actual virus in the executable.

The reason for this alert is the lack of a digital signature. While a code signing certificate would prevent such warnings, I have chosen not to invest in one, as this project is open source (MIT license) and I receive no financial compensation for the work.

#### Download and Run

1. Visit the [Releases page](https://github.com/divergentti/palindrome_game/releases) and download the latest release for your operating system:
   - **Windows**: `PalindromeGame.exe`
   - **Linux**: `PalindromeGame-x86_64.AppImage`
2. Run the executable (after adjusting the Defender):
   - **Windows**: Double-click `PalindromeGame.exe` or run via command line:
     ```bash
     .\PalindromeGame.exe
     ```
   - **Linux**: Make the AppImage executable and run it:
     ```bash
     chmod +x PalindromeGame-x86_64.AppImage
     ./PalindromeGame-x86_64.AppImage
     ```
3. The game will launch with all dependencies and the `palindromes.json` file included, no Python installation required.

**Note**: If you generate new palindromes using `initializer.py`, the updated `palindromes.json` will already be bundled in the binary. To use a different dataset or enable LLM suggestions, you’ll need to rebuild the binary with the updated `palindromes.json` file and API configuration.

### Generating New Palindromes

If you want to generate new palindromes or recreate the dataset, you can use the `initializer.py` script located in the `initializer/` directory:

1. Navigate to the `initializer/` directory:
   ```bash
   cd initializer
   ```
2. Configure `data/runtimeconfig.json` with the appropriate file paths for verbs, nouns, adjectives, and output files. Note that paths should be relative to the `initializer/` directory.
3. Download source data (books, verbs, nouns, adjectives):
   ```bash
   python initializer.py download
   ```
4. Generate palindromes from a specific file (e.g., verbs):
   ```bash
   python initializer.py generate --filename data/english_verbs.csv
   ```
5. Convert generated CSV files to a single JSON file for the game:
   ```bash
   python initializer.py convert
   ```
   This will create `palindromes.json` in the specified directory (as defined in `runtimeconfig.json`). Move the generated `palindromes.json` to the root directory (`../palindromes.json`) for the game to use it.

**Note**: Generating palindromes can take hours to days depending on your hardware and the input size.

## Development Process

### Initial Approach: Random Word Combinations

I started by attempting to generate palindromes by randomly combining English words (verbs, nouns, adjectives). However, this approach was inefficient, as the probability of randomly forming a palindrome is extremely low. Example outputs included:

- "lazy stubborn gravel sway" (not a palindrome)
- "eclectic sinology substitute" (not a palindrome)
- "athletic press murmur" (not a palindrome)

### Building a Palindrome Generator

The `initializer.py` script (in the `initializer/` directory) implements a more structured approach:

1. **Download Data**: Downloads books from Project Gutenberg (e.g., *Pride and Prejudice*, *Moby Dick*) and extracts verbs, nouns, and adjectives using NLTK’s WordNet.
2. **Generate Palindromes**:
   - Starts with a word (e.g., "race").
   - Creates a mirror image (e.g., "racecar") and tests for symmetry.
   - Iterates through the alphabet to insert letters in the middle, forming new symmetrical words (e.g., "racecar" becomes "racecar").
   - Extends palindromes by adding words that maintain symmetry (e.g., "racecar" becomes "racecar level").
   - Validates against vocabulary to ensure meaningfulness.
3. **Result**: Generated over 31,000 palindromes, stored in CSV files (`new_nouns_palindromes.csv`, `new_verbs_palindromes.csv`, `new_adjectives_palindromes.csv`, `new_long_text_palindromes.csv`), and converted to `palindromes.json` for the game, which is now located in the root directory.

### Game Logic

The game logic, implemented in `PalindromeGame.py` (in the root directory), includes:

- **Input Handling**: Uses a `QTextEdit` field for input, with a height of 40 pixels to save space.
- **Highlighting**: Colors characters in real-time (green for matching pairs, red for mismatches, blue for the middle character in odd-length palindromes).
- **Scoring**: Calculates a score based on length and a dummy "sense score" (semantic validity). Known palindromes receive a +5 bonus.
- **Feedback**: Displays whether the input is a palindrome, suggests extensions, and shows similar palindromes from the dataset. LLM-based suggestions provide contextually relevant words to form valid palindromes.
- **Visualization**: A Matplotlib histogram shows the length distribution of pre-generated palindromes, with a marker for the current input’s length.

## Generating Palindromes with `initializer.py`

The `initializer.py` script, located in the `initializer/` directory, is a key component for generating the palindrome dataset:

- **Download Phase**: Downloads books and extracts words (verbs, nouns, adjectives) using NLTK’s WordNet.
- **Generation Phase**:
  - Starts with a word and creates a mirror image.
  - Inserts letters or words to form symmetrical phrases.
  - Validates against vocabulary to ensure meaningfulness.
  - Extends palindromes by adding anagramic words at both ends.
- **Output**: Generated 31,000+ palindromes across four categories, stored in CSV files, and combined into `palindromes.json`, which is placed in the root directory for the game to use.

Example palindromes generated:

- "racecar"
- "level eye level"
- "madam deked madam"

## Challenges and Observations

### Limitations of Machine Learning

- **Symmetry Problem**: ML models like LSTM struggle with the strict symmetry required for palindromes. They excel at predicting probable word sequences but fail to enforce structural constraints.
- **Random Combinations**: Randomly combining words rarely produces palindromes, confirming the need for a structured approach.
- **Sense Validation**: Ensuring palindromes are meaningful (e.g., using real words) required extensive vocabulary checks.

### Player Data Saving Bug (Fixed in v0.0.2)

A critical bug in version 0.0.1 caused player data (total score, playing time, and discovered palindromes) to be reset each time the game was launched. This was due to a global `players = {}` assignment that inadvertently cleared the loaded `players.json` data before the player name check. The issue was resolved in version 0.0.2 by removing the problematic assignment and improving the save logic to merge existing data, ensuring robust persistence across sessions.

### Hybrid Approach

A hybrid model combining ML for word suggestions and logical verification for symmetry proved more effective. The `initializer.py` script uses this approach by suggesting extensions and validating them against word lists. The new LLM-based suggestion feature in version 0.0.2 enhances this by leveraging advanced language models to propose contextually relevant words.

### GUI Development

- **Highlighting Issue**: Initially used `QLineEdit` for input, but it didn’t support HTML-based highlighting. Reverted to `QTextEdit` with a reduced height (40 pixels) to maintain space efficiency.
- **Import Errors**: Fixed issues with PyQt6 (`QAction` moved to `QtGui`) and Matplotlib (`backend_qt6agg` for PyQt6 compatibility).

## Version History

### Version 0.0.2 (22 April 2025)

- **Fixed Player Data Saving**: Resolved a critical bug where player data (total score, playing time, and discovered palindromes) was reset due to a global `players = {}` assignment. The fix ensures `players.json` is loaded and saved correctly, preserving stats across sessions.
- **Added LLM-Based Word Suggestions**: Integrated a large language model (via APIs like LM Studio or Mistral.ai) to suggest words that fit into palindromes. This feature enhances gameplay by guiding players toward valid and meaningful palindromes, accessible via the feedback area or a dedicated button.
- **Improved Save Logic**: Updated `save_players` to merge existing data in `players.json`, preventing accidental overwrites of other players’ stats.

### Version 0.0.1 (Initial Release)

- Initial release with PyQt6 GUI, real-time palindrome checking, scoring, visualization, and pre-generated dataset of over 31,000 palindromes.
- Basic player data saving to `players.json`, with issues in persistence due to global variable bug.

## Future Improvements

- **Enhanced Sense Scoring**: Integrate a more sophisticated semantic analysis (e.g., using NLP models like BERT) to evaluate the "sense" of palindromes.
- **Advanced LLM Integration**: Expand LLM capabilities to generate entire palindrome phrases or provide real-time coaching for complex palindromes.
- **Cross-Platform Builds**: Improve packaging with PyInstaller or cx_Freeze for Windows and Linux distributions, addressing issues with data folder inclusion.
- **Localization**: Add support for other languages beyond English.
- **Game Features**: Introduce levels, challenges, or a multiplayer mode to enhance engagement.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

**Author**: Jari Hiltunen (GitHub: Divergentti)  
**Copyright © 2025**