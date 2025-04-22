# Copyright (c) 2025 Jari Hiltunen / GitHub Divergentti
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import json
import random
import nltk
import os
import matplotlib
import appdirs
import shutil

debug_files = False
debug_gui = False
debug_llm = False


def get_data_dir():
    """Get platform-specific data directory for the app."""
    app_name = "palindromegame"
    app_author = "Divergentti"
    data_dir = appdirs.user_data_dir(app_name, app_author)

    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

# Initialize paths
DATA_DIR = get_data_dir()
PALINDROMES_FILE = os.path.join(DATA_DIR, "palindromes.json")
PLAYERS_FILE = os.path.join(DATA_DIR, "players.json")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")

# For bundled data (Nuitka onefile)
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    BUNDLE_DIR = sys._MEIPASS
else:
    BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PALINDROMES_FILE = os.path.join(BUNDLE_DIR, "palindromes.json")
DEFAULT_PLAYERS_FILE = os.path.join(BUNDLE_DIR, "players.json")
DEFAULT_SETTINGS_FILE = os.path.join(BUNDLE_DIR, "settings.json")

# Load or initialize palindromes
try:
    # 1. Try user's data directory first
    with open(PALINDROMES_FILE, 'r') as f:
        pre_generated = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    try:
        # 2. Fallback to bundled file (for first run)
        with open(DEFAULT_PALINDROMES_FILE, 'r') as f:
            pre_generated = json.load(f)
        # Copy to user directory for future updates
        shutil.copyfile(DEFAULT_PALINDROMES_FILE, PALINDROMES_FILE)
    except Exception as e:
        if debug_files:
            print(f"Error loading palindromes: {e}")
        pre_generated = []
        reverse_pairs = []

# Load player data
try:
    with open(PLAYERS_FILE, 'r') as f:
        players = json.load(f)
    # Normalize player keys to lowercase
    players = {k.lower(): v for k, v in players.items()}
except (FileNotFoundError, json.JSONDecodeError):
    if debug_files:
        print(FileNotFoundError, json.JSONDecodeError)
    try:
        with open(DEFAULT_PLAYERS_FILE, 'r') as f:
            players = json.load(f)
        # Normalize player keys to lowercase
        players = {k.lower(): v for k, v in players.items()}
        shutil.copyfile(DEFAULT_PLAYERS_FILE, PLAYERS_FILE)
    except Exception as e:
        if debug_files:
            print(f"Error loading players: {e}")
        players = {}

if debug_files:
    print(f"Loaded players: {players}")

# Load settings
try:
    with open(SETTINGS_FILE, 'r') as f:
        settings = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    try:
        with open(DEFAULT_SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
        shutil.copyfile(DEFAULT_SETTINGS_FILE, SETTINGS_FILE)
    except Exception as e:
        if debug_files:
            print(f"Error loading settings: {e}")
        settings = {}

def save_players():
    """Save players to platform-specific directory"""
    try:
        existing_players = {}
        if os.path.exists(PLAYERS_FILE):
            with open(PLAYERS_FILE, 'r') as f:
                existing_players = json.load(f)
        normalized_players = {k.lower(): v for k, v in players.items()}
        existing_players.update(normalized_players)
        with open(PLAYERS_FILE, 'w') as f:
            json.dump(existing_players, f, indent=2)
        if debug_files:
            print(f"Saved players: {existing_players}")
    except Exception as e:
        if debug_files:
            print(f"Error saving players: {e}")


if debug_gui:
    print(f"Matplotlib version: {matplotlib.__version__}, Backend: {matplotlib.get_backend()}")
matplotlib.use('QtAgg')
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLineEdit,
    QPushButton, QLabel, QTextEdit, QDialog, QTableWidget, QFormLayout,
    QTableWidgetItem, QMessageBox, QDialogButtonBox, QLCDNumber, QProgressBar, QInputDialog, QFrame, QComboBox
)
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtCore import QTimer, QElapsedTimer, Qt
from matplotlib.figure import Figure

try:
    from matplotlib.backends.backend_qtagg import FigureCanvas

    if debug_gui:
        print("Imported FigureCanvas from backend_qtagg")
except ImportError:
    if debug_gui:
        print("Failed to import FigureCanvas from backend_qtagg, trying backend_qt")
    try:
        from matplotlib.backends.backend_qt import FigureCanvas

        if debug_gui:
            print("Imported FigureCanvas from backend_qt")
    except ImportError:
        if debug_gui:
            print("Failed to import FigureCanvas from backend_qt")
        raise ImportError("Cannot import FigureCanvas; ensure Matplotlib Qt backend is installed")

# Enable Qt debugging
os.environ["QT_LOGGING_RULES"] = "qt6.*=true"

# Download NLTK words if not already downloaded
NLTK_DATA_DIR = os.path.join(DATA_DIR, 'nltk_data')
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path = [NLTK_DATA_DIR]
os.environ['NLTK_DATA'] = NLTK_DATA_DIR

try:
    nltk.data.find('corpora/words')
    if debug_files:
        print("NLTK words already downloaded")
except LookupError:
    if debug_files:
        print(f"Downloading NLTK words to {NLTK_DATA_DIR}")
    nltk.download('words', download_dir=NLTK_DATA_DIR, quiet=True)

from nltk.corpus import words
english_words = set(words.words())
reverse_pairs = [(w, w[::-1]) for w in english_words if w[::-1] in english_words and w < w[::-1]]

GAME_VERSION = "Ver. 0.0.2 - 22.04.2025"

class LargeLanguageModelsAPI:
    def __init__(self, query_prompt, system_prompt=None, max_tokens=200, backend="lm_studio", temperature=0.7, conversation=None):
        self.system_prompt = system_prompt or (
            "You are a professional palindrome innovator. "
            "Provide concise suggestions for words that fit into the user's palindrome."
            "Response must contain only words separated by commas."
            "Do not repeat the system prompt or provide lengthy explanations."
        )
        self.query_prompt = query_prompt
        self.max_tokens = max_tokens
        self.backend = backend
        self.temperature = temperature
        self.conversation = conversation

    def query(self):
        if not self.conversation:
            return "Error: No conversation instance provided."

        if self.backend == "lm_studio":
            return self._query_lm_studio()
        elif self.backend == "mistral":
            return self._query_mistral()
        elif self.backend == "openai":
            return self._query_openai()
        else:# Download NLTK words if not already downloaded
            return f"Unsupported backend selected: {self.backend}"

    def _build_prompt(self):
        return f"{self.system_prompt}\n\nUser prompt:\n{self.query_prompt}"

    def _query_lm_studio(self):
        return self.conversation.query_lm_studio(
            prompt=self._build_prompt()
        )

    def _query_mistral(self):
        return self.conversation.query_mistral(
            prompt=self._build_prompt(),
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

    def _query_openai(self):
        return self.conversation.query_openai(
            system_prompt=self.system_prompt,
            user_prompt=self.query_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

class LLMConversation:
    def __init__(self, lm_studio_url=None, mistral_api_key=None, openai_api_key=None):
        self.lm_studio_url = lm_studio_url or "http://localhost:1234/v1/chat/completions"
        self.mistral_api_key = mistral_api_key
        self.openai_api_key = openai_api_key

    def query_lm_studio(self, prompt):
        import requests
        payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.7,
        "stream": False  }
        try:
            response = requests.post(self.lm_studio_url, json=payload)
            response.raise_for_status()
            if debug_llm:
                print(response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip())
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        except Exception as e:
            if debug_llm:
                print(f"LM Studio error: {str(e)}")
            return f"LM Studio error: {str(e)}"

    def query_mistral(self, prompt, max_tokens=200, temperature=0.7):
        import requests
        headers = {
            "Authorization": f"Bearer {self.mistral_api_key}",
            "Content-Type": "application/json",
            "mistral-version": "2024-05-10"
        }
        payload = {
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        try:
            response = requests.post("https://api.mistral.ai/v1/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            if debug_llm:
                print(content)

            suggestions = content.replace("<ul>", "").replace("</ul>", "").replace("<li>", "").replace("</li>", ", ")
            suggestions = suggestions.strip(", ")
            return suggestions
        except Exception as e:
            if debug_llm:
                print(f"Mistral error: {str(e)}")
            return f"Mistral error: {str(e)}"

    def query_openai(self, system_prompt, user_prompt, max_tokens=200, temperature=0.7):
        import openai
        openai.api_key = self.openai_api_key
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            content = response["choices"][0]["message"]["content"].strip()
            if debug_llm:
                print(content)

            suggestions = content.replace("<ul>", "").replace("</ul>", "").replace("<li>", "").replace("</li>", ", ")
            suggestions = suggestions.strip(", ")
            return suggestions
        except Exception as e:
            if debug_llm:
                print(f"OpenAI error: {str(e)}")
            return f"OpenAI error: {str(e)}"


class SettingsManager:
    def __init__(self):
        self.settings = {}
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f:
                self.settings = json.load(f)

    def get_conversation(self):
        return LLMConversation(
            lm_studio_url=self.settings.get("lm_studio_url"),
            mistral_api_key=self.settings.get("mistral_api_key"),
            openai_api_key=self.settings.get("openai_api_key")
        )

    def get_backend(self):
        return self.settings.get("backend", "lm_studio")


class SettingsDialog(QDialog):
    """Dialog for selecting LLM backend and entering API keys."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LLM Settings")
        self.setGeometry(200, 200, 500, 300)
        layout = QVBoxLayout()

        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f7;
            }
            QLineEdit, QComboBox {
                padding: 8px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                font-size: 12pt;
            }
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
        """)

        form_layout = QFormLayout()

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["lm_studio", "mistral", "openai"])
        self.backend_combo.currentTextChanged.connect(self.update_visibility)
        form_layout.addRow("Backend:", self.backend_combo)

        self.lm_studio_url_input = QLineEdit()
        form_layout.addRow("LM Studio URL:", self.lm_studio_url_input)

        self.mistral_key_input = QLineEdit()
        self.mistral_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        form_layout.addRow("Mistral API Key:", self.mistral_key_input)

        self.openai_key_input = QLineEdit()
        self.openai_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        form_layout.addRow("OpenAI API Key:", self.openai_key_input)

        layout.addLayout(form_layout)

        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(self.save_settings)
        layout.addWidget(save_button)

        self.setLayout(layout)
        self.load_settings()
        self.update_visibility()

    def update_visibility(self):
        backend = self.backend_combo.currentText()
        self.lm_studio_url_input.setVisible(backend == "lm_studio")
        self.mistral_key_input.setVisible(backend == "mistral")
        self.openai_key_input.setVisible(backend == "openai")

    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
                self.backend_combo.setCurrentText(settings.get("backend", "lm_studio"))
                self.lm_studio_url_input.setText(settings.get("lm_studio_url", ""))
                self.mistral_key_input.setText(settings.get("mistral_api_key", ""))
                self.openai_key_input.setText(settings.get("openai_api_key", ""))

    def save_settings(self):
        settings = {
            "backend": self.backend_combo.currentText(),
            "lm_studio_url": self.lm_studio_url_input.text(),
            "mistral_api_key": self.mistral_key_input.text(),
            "openai_api_key": self.openai_key_input.text()
        }
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
        QMessageBox.information(self, "Saved", "Settings saved successfully.")
        self.accept()


class Inspector:
    """Inspects if user input is a palindrome and calculates its 'make sense' score."""

    def is_symmetric(self, text):
        cleaned = ''.join(text.split()).lower()
        return cleaned == cleaned[::-1]

    def make_sense_score(self, phrase):
        words_in_phrase = phrase.split()
        if not words_in_phrase:
            return 0
        valid_words = [word for word in words_in_phrase if word.lower() in english_words]
        return (len(valid_words) / len(words_in_phrase)) * 100

    def inspect(self, text):
        symmetric = self.is_symmetric(text)
        score = self.make_sense_score(text) if symmetric else 0
        return symmetric, score


class Suggestor:
    """Provides suggestions to extend a palindrome."""

    def suggest_extensions(self, palindrome, max_suggestions=5):
        if not reverse_pairs:
            return []
        suggestions = []
        num_suggestions = min(max_suggestions, len(reverse_pairs))
        selected_pairs = random.sample(reverse_pairs, num_suggestions)
        for w, r in selected_pairs:
            extended = f"{w} {palindrome} {r}"
            suggestions.append(extended)
        return suggestions


class GameInstructionsDialog(QDialog):
    """Dialog to display game instructions."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Game Instructions")
        self.setGeometry(100, 100, 500, 600)
        layout = QVBoxLayout()
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f7;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 2px;
                padding: 10px;
                font-size: 11pt;
            }
        """)

        instructions = QTextEdit()
        instructions.setReadOnly(True)
        instructions.setText(
            "Welcome to the Palindrome Game!\n\n"
            "How to Play:\n"
            "1. Type your text in the input field (special characters are not allowed).\n"
            "2. Once your text exceeds 5 characters, it will be checked to see if it's a palindrome.\n"
            "3. Earn points for valid palindromes based on:\n"
            "   - Length of the palindrome\n"
            "   - Meaningfulness (how much 'sense' it makes)\n"
            "   - +5 bonus points for known palindromes\n"
            "4. New palindromes will be saved for future games.\n\n"
            "Scoring Details:\n"
            "- Base Score: (Length × Meaningfulness Percentage) / 100\n"
            "- Bonus Points: +5 for known palindromes\n\n"
            "Visual Feedback:\n"
            "- Symmetry and progress indicators\n"
            "- A light blue character marks the middle point (mirror from here)\n\n"
            "Setting Up LLM:\n"
            "- Install LM Studio and download the appropriate model, or\n"
            "- Obtain API keys from a provider (e.g., OpenAI or Mistral.ai)\n\n"
            "Enjoy creating word art!\n\n"
            "(C) 2025 - Jari Hiltunen - Licensed under MIT"
        )

        font = QFont("Arial", 10)
        instructions.setFont(font)

        layout.addWidget(instructions)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
        """)
        layout.addWidget(close_button)
        self.setLayout(layout)


class PlayerName(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Who is playing today?")
        self.setGeometry(100, 100, 350, 150)
        layout = QVBoxLayout()
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f7;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                font-size: 14pt;
            }
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
        """)
        welcome_label = QLabel("Welcome to the Palindrome Game!")
        welcome_label.setStyleSheet("font-size: 14pt; font-weight: bold; margin-bottom: 10px;")
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(welcome_label)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter your name")
        layout.addWidget(self.name_input)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def get_player_name(self):
        name = self.name_input.text().strip().lower()
        if not name or not name.isalnum():
            if debug_files:
                print(f"PlayerName.get_player_name: Invalid name '{name}'")
            return ""
        if debug_files:
            print(f"PlayerName.get_player_name: returning '{name}'")
        return name


class InspectPalindromesDialog(QDialog):
    """Dialog to inspect existing palindromes with search functionality."""

    def __init__(self, palindromes):
        super().__init__()
        self.setWindowTitle("Inspect Palindromes")
        self.setGeometry(100, 100, 800, 500)
        self.palindromes = palindromes
        layout = QVBoxLayout()
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f7;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #cccccc;
                border-radius: 2px;
                font-size: 11pt;
            }
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 1px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
            QTableWidget {
                background-color: white;
                alternate-background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-radius: 2px;
                gridline-color: #e0e0e0;
            }
            QHeaderView::section {
                background-color: #e0e0e0;
                padding: 5px;
                border: 1px solid #cccccc;
                font-weight: bold;
            }
        """)

        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search palindromes...")
        self.search_input.textChanged.connect(self.filter_palindromes)
        search_layout.addWidget(self.search_input)
        search_button = QPushButton("Search")
        search_button.clicked.connect(self.filter_palindromes)
        search_layout.addWidget(search_button)
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear_search)
        search_layout.addWidget(clear_button)
        layout.addLayout(search_layout)

        self.counter_label = QLabel(f"Showing all {len(palindromes)} palindromes")
        self.counter_label.setStyleSheet("font-size: 12pt; margin: 10px 0;")
        layout.addWidget(self.counter_label)

        self.table = QTableWidget()
        self.table.setColumnCount(1)
        self.table.setHorizontalHeaderLabels(["Palindrome"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setFont(QFont("Arial", 12))
        self.populate_table(palindromes)
        layout.addWidget(self.table)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

        self.setLayout(layout)

    def populate_table(self, palindromes_to_show):
        self.table.setRowCount(len(palindromes_to_show))
        for i, p in enumerate(palindromes_to_show):
            item = QTableWidgetItem(p)
            # Make items more readable
            item.setFont(QFont("Arial", 12))
            self.table.setItem(i, 0, item)
        self.counter_label.setText(f"Showing {len(palindromes_to_show)} of {len(self.palindromes)} palindromes")

    def filter_palindromes(self):
        search_text = self.search_input.text().lower()
        if not search_text:
            self.populate_table(self.palindromes)
            return
        matching_palindromes = [p for p in self.palindromes if search_text in p.lower()]
        self.populate_table(matching_palindromes)
        if matching_palindromes:
            self.counter_label.setText(
                f"Found {len(matching_palindromes)} matching palindromes out of {len(self.palindromes)}")
        else:
            self.counter_label.setText("No matching palindromes found")

    def clear_search(self):
        self.search_input.clear()
        self.populate_table(self.palindromes)


class PalindromeVisualizationDialog(QDialog):
    """Dialog to visualize palindrome lengths using Matplotlib."""

    def __init__(self, palindromes):
        super().__init__()
        self.setWindowTitle("Palindrome Length Distribution")
        self.setGeometry(100, 100, 700, 500)
        layout = QVBoxLayout()
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f7;
            }
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 2px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
        """)

        if debug_gui:
            print(f"PalindromeVisualizationDialog: {len(palindromes)} palindromes received")
        lengths = [len(p.replace(" ", "")) for p in palindromes]
        if not lengths:
            if debug_gui:
                print("PalindromeVisualizationDialog: No palindrome lengths to visualize")
            QMessageBox.information(self, "No Data", "No palindromes to visualize.")
            self.reject()  # Close the dialog
            return
        try:
            fig = Figure(figsize=(8, 6), dpi=100)
            fig.patch.set_facecolor('#f5f5f7')  # Match dialog background
            ax = fig.add_subplot(111)

            n, bins, patches = ax.hist(lengths, bins=20, edgecolor='white', alpha=0.8, color='#4a86e8')
            ax.set_title("Distribution of Palindrome Lengths", fontsize=14, fontweight='bold')
            ax.set_xlabel("Length", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)

            # Customize grid and spines for a modern look
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            canvas = FigureCanvas(fig)
            canvas.draw()  # Explicitly draw the canvas
            layout.addWidget(canvas)

            if debug_gui:
                print("PalindromeVisualizationDialog: Matplotlib canvas added to layout and drawn")
        except Exception as e:
            if debug_gui:
                print(f"PalindromeVisualizationDialog: Error creating plot: {e}")
            QMessageBox.critical(self, "Error", f"Failed to create visualization: {str(e)}")
            self.reject()  # Close the dialog
            return

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

        self.setLayout(layout)
        if debug_gui:
            print("PalindromeVisualizationDialog: Dialog layout set")
        self.show()


class PlayerStatsDialog(QDialog):
    def __init__(self, player_name, player_data):
        super().__init__()
        print(f"PlayerStatsDialog: player_name={player_name}, player_data={player_data}")
        self.setWindowTitle(f"Stats for {player_name}")
        self.setGeometry(100, 100, 500, 400)
        layout = QVBoxLayout()
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f7;
            }
            QLabel {
                font-size: 12pt;
                margin: 5px 0;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 2px;
                padding: 10px;
                font-size: 11pt;
            }
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 2px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
        """)
        player_header = QLabel(f"Player: {player_name}")
        player_header.setStyleSheet("font-size: 18pt; font-weight: bold; margin-bottom: 15px;")
        player_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(player_header)
        stats_frame = QFrame()
        stats_frame.setFrameShape(QFrame.Shape.StyledPanel)
        stats_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 4px;
                padding: 2px;
            }
        """)
        stats_layout = QVBoxLayout(stats_frame)
        score_label = QLabel(f"Total Score: {player_data.get('total_score', 0.0):.2f}")
        stats_layout.addWidget(score_label)
        playing_time = player_data.get('playing_time', 0)
        if playing_time < 60:
            time_label = QLabel(f"Playing Time: {playing_time:.2f} seconds")
        else:
            time_label = QLabel(f"Playing Time: {playing_time / 3600:.2f} hours")
        stats_layout.addWidget(time_label)
        layout.addWidget(stats_frame)
        palindromes_label = QLabel("New Palindromes Found:")
        palindromes_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(palindromes_label)
        palindromes_list = QTextEdit()
        palindromes_list.setReadOnly(True)
        palindromes_text = "\n".join(player_data.get('new_palindromes', [])) or "No new palindromes found."
        palindromes_list.setText(palindromes_text)
        layout.addWidget(palindromes_list)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)
        self.setLayout(layout)


class MainWindow(QMainWindow):
    """Main window for the Palindrome Game."""

    def __init__(self, player, player_name):
        super().__init__()
        self.setWindowTitle("Palindrome Game " + GAME_VERSION)
        self.setGeometry(100, 100, 900, 700)
        self.player = player
        self.player_name = player_name
        self.inspector = Inspector()
        self.suggestor = Suggestor()
        self.settings = {"num_matches": 5}
        self.timer = QElapsedTimer()
        self.timer.start()
        self.highlight_timer = QTimer()
        self.highlight_timer.setSingleShot(True)
        self.highlight_timer.timeout.connect(self._update_highlighting)

        # Set application-wide font
        app_font = QFont("Arial", 11)
        self.setFont(app_font)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f7;
            }
            QMenuBar {
                background-color: #333333;
                color: white;
                padding: 2px;
                font-size: 11pt;
            }
            QMenuBar::item {
                padding: 2px 10px;
            }
            QMenuBar::item:selected {
                background-color: #555555;
            }
            QMenu {
                background-color: #333333;
                color: white;
                border: 1px solid #555555;
            }
            QMenu::item:selected {
                background-color: #4a86e8;
            }
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 2px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
            QLabel {
                font-size: 11pt;
            }
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 4px;
                text-align: center;
                height: 20px;
                font-size: 11pt;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4a86e8;
                border-radius: 3px;
            }
        """)

        # Menu bar
        menubar = self.menuBar()
        help_menu = menubar.addMenu("Help")
        instructions_action = QAction("Game Instructions", self)
        instructions_action.triggered.connect(self.show_instructions)
        help_menu.addAction(instructions_action)

        inspect_menu = menubar.addMenu("Inspect")
        inspect_action = QAction("Inspect Palindromes", self)
        inspect_action.triggered.connect(self.show_inspect_palindromes)
        inspect_menu.addAction(inspect_action)
        viz_action = QAction("Visualize Palindromes", self)
        viz_action.triggered.connect(self.show_visualization)
        inspect_menu.addAction(viz_action)

        player_menu = menubar.addMenu("Player")
        stats_action = QAction("View Stats", self)
        stats_action.triggered.connect(self.show_player_stats)
        player_menu.addAction(stats_action)

        settings_menu = menubar.addMenu("Settings")
        num_matches_action = QAction("Set Number of Matching Palindromes", self)
        num_matches_action.triggered.connect(self.set_num_matches)
        settings_menu.addAction(num_matches_action)

        settings_action = QAction("LLM Settings", self)
        settings_action.triggered.connect(self.show_llm_settings)
        settings_menu.addAction(settings_action)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Player info header
        player_info = QLabel(f"Current Player: {player_name} | Total Score: {player['total_score']:.2f}")
        player_info.setObjectName("playerInfoLabel")
        player_info.setStyleSheet("font-size: 12pt; font-weight: bold; margin: 10px 0; color: #333333;")
        player_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(player_info)

        # Input field with reduced height
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("Enter a phrase to create a palindrome...")
        self.input_field.setMaximumHeight(40)
        self.input_field.setStyleSheet("""
            QTextEdit {
                background-color: white;
                color: #333333;
                border: 2px solid #cccccc;
                border-radius: 2px;
                padding: 5px;
                font-size: 12pt;
            }
        """)
        self.input_field.textChanged.connect(self.on_text_changed)
        layout.addWidget(self.input_field)
        lcd_layout = QHBoxLayout()
        lcd_frame = QFrame()
        lcd_frame.setFrameShape(QFrame.Shape.StyledPanel)
        lcd_frame.setStyleSheet("""
            QFrame {
                background-color: #333333;
                border-radius: 4px;
                padding: 2px;
            }
            QLabel {
                color: white;
                font-weight: bold;
            }
        """)
        lcd_inner_layout = QHBoxLayout(lcd_frame)

        # Shared font for LCDs
        lcd_font = QFont()
        lcd_font.setPointSize(10)

        # Length LCD
        length_layout = QVBoxLayout()
        length_label = QLabel("Length:")
        length_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        length_layout.addWidget(length_label)
        self.length_lcd = QLCDNumber()
        self.length_lcd.setDigitCount(3)
        self.length_lcd.setSegmentStyle(QLCDNumber.SegmentStyle.Filled)
        self.length_lcd.setFont(lcd_font)
        self.length_lcd.setFixedHeight(30)
        self.length_lcd.setStyleSheet("""
            QLCDNumber {
                background-color: #222222;
                color: #00ff00;
                border: 2px solid #444444;
                border-radius: 2px;
            }
        """)
        length_layout.addWidget(self.length_lcd)
        lcd_inner_layout.addLayout(length_layout)

        # Sense LCD
        sense_layout = QVBoxLayout()
        sense_label = QLabel("Sense Score:")
        sense_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sense_layout.addWidget(sense_label)
        self.sense_lcd = QLCDNumber()
        self.sense_lcd.setDigitCount(5)
        self.sense_lcd.setSegmentStyle(QLCDNumber.SegmentStyle.Filled)
        self.sense_lcd.setFont(lcd_font)
        self.sense_lcd.setFixedHeight(30)
        self.sense_lcd.setStyleSheet("""
            QLCDNumber {
                background-color: #222222;
                color: #00ff00;
                border: 2px solid #444444;
                border-radius: 2px;
            }
        """)
        sense_layout.addWidget(self.sense_lcd)
        lcd_inner_layout.addLayout(sense_layout)

        # Score LCD
        score_layout = QVBoxLayout()
        score_label = QLabel("Total Score:")
        score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_layout.addWidget(score_label)
        self.score_lcd = QLCDNumber()
        self.score_lcd.setDigitCount(8)
        self.score_lcd.setSegmentStyle(QLCDNumber.SegmentStyle.Filled)
        self.score_lcd.display(self.player["total_score"])
        self.score_lcd.setFont(lcd_font)
        self.score_lcd.setFixedHeight(30)
        self.score_lcd.setStyleSheet("""
            QLCDNumber {
                background-color: #222222;
                color: #33ff33;
                border: 2px solid #444444;
                border-radius: 2px;
            }
        """)
        score_layout.addWidget(self.score_lcd)
        lcd_inner_layout.addLayout(score_layout)

        lcd_layout.addWidget(lcd_frame)
        layout.addLayout(lcd_layout)

        # Progress bar
        progress_label = QLabel("Palindrome Completeness:")
        progress_label.setStyleSheet("font-size: 12pt; font-weight: bold; margin-top: 10px;")
        layout.addWidget(progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 2px;
                background-color: #f0f0f0;
                text-align: center;
                color: black;
                font-weight: bold;
                height: 10px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #5a96f8, stop:1 #4a86e8);
                border-radius: 2px;
            }
        """)
        layout.addWidget(self.progress_bar)

        feedback_label = QLabel("Feedback:")
        feedback_label.setStyleSheet("font-size: 12pt; font-weight: bold; margin-top: 10px;")
        layout.addWidget(feedback_label)

        self.feedback_area = QTextEdit()
        self.feedback_area.setReadOnly(True)
        self.feedback_area.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 10px;
                font-size: 12pt;
                line-height: 1.4;
            }
        """)
        layout.addWidget(self.feedback_area)
        self.llm_button = QPushButton("Ask LLM for Feedback")
        self.llm_button.clicked.connect(self.ask_llm_feedback)
        layout.addWidget(self.llm_button)

        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        divider.setStyleSheet("background-color: #cccccc; margin: 10px 0;")
        layout.addWidget(divider)

        # Matplotlib visualization
        viz_label = QLabel("Palindrome Length Distribution:")
        viz_label.setStyleSheet("font-size: 12pt; font-weight: bold; margin-top: 5px;")
        layout.addWidget(viz_label)

        self.corpus_lengths = [len(p.replace(" ", "")) for p in pre_generated]
        self.fig = Figure(figsize=(8, 3), dpi=100)
        self.fig.patch.set_facecolor('#f8f8f8')
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        self.update_visualization()

    def show_llm_settings(self):
        dialog = SettingsDialog(self)
        dialog.exec()

    def _update_highlighting(self):
        """Update text highlighting for the input field."""
        text = self.input_field.toPlainText()
        if debug_gui:
            print(f"_update_highlighting: Text='{text}', Length={len(text)}, Spaces={text.count(' ')}")
        positions = [i for i, c in enumerate(text) if not c.isspace()]
        if not positions:
            self.input_field.blockSignals(True)
            self.input_field.setHtml(text)
            self.input_field.blockSignals(False)
            if debug_gui:
                print("_update_highlighting: Empty input, HTML set")
            return

        cleaned = ''.join([text[i] for i in positions]).lower()
        length = len(cleaned)
        colors = ['white'] * len(text)

        for idx, pos in enumerate(positions):
            if idx < length // 2:
                if cleaned[idx] == cleaned[-idx - 1]:
                    colors[pos] = '#a3ffb3'  # Light green
                else:
                    colors[pos] = '#ffb3b3'  # Light red
                sym_pos = positions[-idx - 1]
                colors[sym_pos] = colors[pos]
            elif length % 2 == 1 and idx == length // 2:
                colors[pos] = '#b3d9ff'  # Light blue

        html = ""
        for i, char in enumerate(text):
            if i in positions:
                text_color = "black" if colors[i] != '#ffb3b3' else "#333333"
                html += f'<span style="background-color: {colors[i]}; color: {text_color};">{char}</span>'
            else:
                html += char

        self.input_field.blockSignals(True)
        cursor = self.input_field.textCursor()
        position = cursor.position()
        if debug_gui:
            print(f"_update_highlighting: Before setHtml, Cursor position={position}, HTML='{html}'")
        self.input_field.setHtml(html)
        current_text = self.input_field.toPlainText()
        max_position = len(current_text)
        if position > max_position:
            if debug_gui:
                print(f"_update_highlighting: Adjusting cursor position from {position} to {max_position}")
            position = max_position
        cursor.setPosition(position)
        self.input_field.setTextCursor(cursor)
        self.input_field.blockSignals(False)
        if debug_gui:
            print(f"_update_highlighting: After setHtml, Cursor position={cursor.position()}")

    def update_progress_bar(self, text):
        """Update the progress bar based on palindrome progress."""
        positions = [i for i, c in enumerate(text) if not c.isspace()]
        if not positions:
            self.progress_bar.setValue(0)
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #cccccc;
                    border-radius: 6px;
                    background-color: #f0f0f0;
                    text-align: center;
                    color: black;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: #4a86e8;
                    border-radius: 5px;
                }
            """)
            return

        cleaned = ''.join([text[i] for i in positions]).lower()
        length = len(cleaned)
        if length == 0:
            self.progress_bar.setValue(0)
            return

        matches = sum(1 for i in range(length // 2) if cleaned[i] == cleaned[-i - 1])
        total_pairs = length // 2
        percentage = 100 if total_pairs == 0 else (matches / total_pairs) * 100
        self.progress_bar.setValue(int(percentage))

        if percentage < 50:
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #cccccc;
                    border-radius: 6px;
                    background-color: #f0f0f0;
                    text-align: center;
                    color: black;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: #ff9966;
                    border-radius: 5px;
                }
            """)
        elif percentage < 100:
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #cccccc;
                    border-radius: 6px;
                    background-color: #f0f0f0;
                    text-align: center;
                    color: black;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: #ffcc66;
                    border-radius: 5px;
                }
            """)
        else:
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #cccccc;
                    border-radius: 6px;
                    background-color: #f0f0f0;
                    text-align: center;
                    color: black;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: #66cc66;
                    border-radius: 5px;
                }
            """)

    def update_visualization(self):
        """Update the Matplotlib visualization of palindrome lengths."""
        self.ax.clear()
        n, bins, patches = self.ax.hist(self.corpus_lengths, bins=20, edgecolor='white', alpha=0.7,
                                        color='#4a86e8', rwidth=0.85)
        self.ax.set_title("Corpus Palindrome Length Distribution", fontsize=12, fontweight='bold')
        self.ax.set_xlabel("Length", fontsize=10)
        self.ax.set_ylabel("Frequency", fontsize=10)
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        current_text = self.input_field.toPlainText().strip()
        current_length = len(current_text.replace(" ", ""))
        if current_length > 0:
            self.ax.axvline(x=current_length, color='#ff5555', linestyle='--',
                            linewidth=2, label=f'Current: {current_length}')
            self.ax.legend(loc='upper right', frameon=True, fontsize=10)

        self.fig.tight_layout()
        self.canvas.draw()
        if debug_gui:
            print(f"update_visualization: Updated plot, current length={current_length}")

    def check_palindrome(self):
        """Check if the input is a palindrome and provide feedback."""
        text = self.input_field.toPlainText().strip()
        if len(text) <= 5:
            return

        is_palindrome, sense_score = self.inspector.inspect(text)
        self.length_lcd.display(len(text))
        self.sense_lcd.display(sense_score)

        if is_palindrome:
            length = len(text)

            if text in pre_generated:
                total_score = length + 5
                feedback = (
                    f"<b>SUCCESS!</b> '{text}' is a <b>known palindrome</b>.<br>"
                    f"<b>Score:</b> {total_score:.2f} points (+5 bonus).<br>"
                    f"<b>Sense score:</b> {sense_score:.1f}%"
                )
            else:
                total_score = length * sense_score
                pre_generated.append(text)
                if debug_files:
                    print(f"Appended palindrome {text}")
                with open(PALINDROMES_FILE, 'w') as f:
                    json.dump(pre_generated, f)
                    if debug_files:
                        print(f"File {PALINDROMES_FILE} saved.")
                self.player["new_palindromes"].append(text)
                feedback = (
                    f"<b>SUCCESS!</b> '{text}' is a <b>new palindrome</b>!<br>"
                    f"<b>Score:</b> {total_score:.2f} points<br>"
                    f"<b>Sense score:</b> {sense_score:.1f}%<br>"
                    f"<b>Added to your collection!</b>"
                )

            self.player["total_score"] += total_score
            self.score_lcd.display(self.player["total_score"])

            suggestions = self.suggestor.suggest_extensions(text)
            if suggestions:
                feedback += "<b> Try these extensions:</b><ul>"
                for i, s in enumerate(suggestions, 1):
                    feedback += f"<li>{s}</li>"
                feedback += "</ul>"

            self.update_player_info()

        else:
            feedback = f"<span style='color: #cc3333;'><b>Not a palindrome yet.</b></span> '{text}' does not read the same backward.<br>"
            cleaned = ''.join(text.split()).lower()
            reverse = cleaned[::-1]
            match_count = sum(1 for a, b in zip(cleaned, reverse) if a == b)
            if cleaned:
                match_percentage = (match_count / len(cleaned)) * 100
                feedback += f"<b>Match:</b> {match_percentage:.1f}% of characters match their mirror position."

            matches = [p for p in pre_generated if cleaned in ''.join(p.split()).lower()]
            if matches:
                num_to_show = self.settings["num_matches"]
                feedback += f"<b> Similar palindromes</b> ({min(num_to_show, len(matches))} of {len(matches)}):<ul>"
                for i, m in enumerate(matches[:num_to_show], 1):
                    feedback += f"<li>{m}</li>"
                feedback += "</ul>"

        self.feedback_area.setHtml(feedback)

    def update_player_info(self):
        """Update the player information display."""
        player_info = self.findChild(QLabel, "playerInfoLabel")
        if player_info:
            player_info.setText(f"Current Player: {self.player_name} | Total Score: {self.player['total_score']:.2f}")

    def on_text_changed(self):
        """Handle text changes in the input field."""
        text = self.input_field.toPlainText().strip()

        # Check for invalid characters (only letters and spaces allowed)
        invalid_chars = [c for c in text if not c.isalpha() and not c.isspace()]
        if invalid_chars:
            unique_chars = ", ".join(f"'{c}'" for c in sorted(set(invalid_chars)))
            self.feedback_area.setHtml(
                f'<span style="color: #cc3333;"><b>Warning:</b> Invalid characters detected: {unique_chars}<br>'
                'Only letters (A-Z, a-z) and spaces are allowed!</span>'
            )
        else:
            if len(text) <= 5:
                self.feedback_area.clear()

        if debug_gui:
            print(f"on_text_changed: Text='{text}', Length={len(text)}, Spaces={text.count(' ')}")

        self.update_progress_bar(text)
        self.highlight_timer.stop()
        self.highlight_timer.start(300)

        if hasattr(self, 'check_timer'):
            self.check_timer.stop()

        self.check_timer = QTimer()
        self.check_timer.setSingleShot(True)
        self.check_timer.timeout.connect(self.check_palindrome)
        self.check_timer.start(500)
        self.update_visualization()

    def ask_llm_feedback(self):
        """Ask the selected LLM for palindrome suggestions."""
        user_input = self.input_field.toPlainText().strip()
        if len(user_input) <= 3:
            self.feedback_area.append("<i>Enter more text before asking LLM.</i>")
            return

        self.llm_button.setEnabled(False)
        self.llm_button.setText("Waiting...")

        try:
            settings = SettingsManager()
            conversation = settings.get_conversation()
            backend = settings.get_backend()

            llm_api = LargeLanguageModelsAPI(
                query_prompt=user_input,
                conversation=conversation,
                backend=backend
            )

            response = llm_api.query()

            # Reset button
            self.llm_button.setEnabled(True)
            self.llm_button.setText("Ask LLM for Suggestions")

            if not response:
                self.feedback_area.append("<i>No response from LLM.</i>")
                return

            suggestions = self._parse_llm_suggestions(response)
            if suggestions:
                self.feedback_area.append("<b>LLM Suggestions:</b>")
                for s in suggestions:
                    self.feedback_area.append(f" {s}")
            else:
                self.feedback_area.append(f"<b>LLM Response:</b> {response}")

        except Exception as e:
            self.feedback_area.append(f"<span style='color: red;'>LLM error: {str(e)}</span>")
            self.llm_button.setEnabled(True)
            self.llm_button.setText("Ask LLM for Feedback")

    def _parse_llm_suggestions(self, response: str) -> list[str]:
        """Try to extract a list of suggestions from LLM response text."""
        lines = response.splitlines()
        suggestions = []

        for line in lines:
            line = line.strip("-•*1234567890. \t")  # Remove bullet/list characters
            if len(line.split()) >= 2 and len(line) > 5:
                suggestions.append(line.strip())
        return suggestions

    def show_instructions(self):
        dialog = GameInstructionsDialog()
        dialog.exec()

    def show_inspect_palindromes(self):
        dialog = InspectPalindromesDialog(pre_generated)
        dialog.exec()

    def show_visualization(self):
        dialog = PalindromeVisualizationDialog(pre_generated)
        dialog.exec()

    def show_player_stats(self):
        if debug_gui:
            print(f"show_player_stats: player_name={self.player_name}, player_data={self.player}")
        dialog = PlayerStatsDialog(self.player_name, self.player)
        dialog.exec()

    def set_num_matches(self):
        num, ok = QInputDialog.getInt(self, "Set Number", "Number of matching palindromes to show:", value=5, min=1, max=100)
        if ok:
            self.settings["num_matches"] = num

    def closeEvent(self, event):
        elapsed = self.timer.elapsed() / 1000
        # Update playing_time in self.player
        original_time = self.player["playing_time"]
        self.player["playing_time"] += elapsed
        # Update global players dictionary with the latest self.player data
        players[self.player_name] = {
            "total_score": self.player.get("total_score", 0.0),
            "playing_time": self.player["playing_time"],
            "new_palindromes": self.player.get("new_palindromes", [])
        }
        if debug_files:
            print(
                f"Closing app - Adding {elapsed:.2f} seconds to player time "
                f"(was {original_time:.2f}, now {self.player['playing_time']:.2f})"
            )
        try:
            save_players()
            if debug_files:
                print(f"Players data saved successfully to {PLAYERS_FILE}")
        except Exception as e:
            if debug_files:
                print(f"Error saving players data: {e}")
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app_font = QFont("Arial", 11)
    QApplication.setFont(app_font)

    if debug_files:
        print(f"Before PlayerName dialog: players={players}")

    player_dialog = PlayerName()
    if player_dialog.exec() == QDialog.DialogCode.Accepted:
        player_name = player_dialog.get_player_name()
        if debug_files:
            print(f"After PlayerName dialog: player_name='{player_name}', players={players}")
        if player_name:
            # If player doesn't exist, create a new entry and save
            if player_name not in players:
                players[player_name] = {"total_score": 0.0, "playing_time": 0, "new_palindromes": []}
                if debug_files:
                    print(f"Created new player: {player_name}")
                save_players()  # Save only for new players
            else:
                if debug_files:
                    print(f"Using existing player: {player_name}, data={players[player_name]}")
            # Start the game with the player's data
            print(f"Passing to MainWindow: player_name={player_name}, player_data={players[player_name]}")
            window = MainWindow(players[player_name], player_name)
            window.show()
            sys.exit(app.exec())
        else:
            QMessageBox.warning(None, "No Name", "Please enter a valid name.")
            sys.exit(1)
    else:
        sys.exit(0)