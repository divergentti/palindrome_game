# Copyright (c) 2025 Jari Hiltunen / GitHub Divergentti
#
#
# Palindrome Game Initializer - Version 0.0.1 - 15.05.2025
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
#
# Downloading and initializing words for the game is time consuming process.
# Use this script to download English books and words and for ML-model initialization.
# Typically you do not need this script at all if you like predefined data in the game!
#
# Arguments: download, generate [filename]
# - download will download books to a txt-file, verbs, nouns and adjectives to csv-files
# - generate will start palindrome (3 words) generation, [filename] is csv-file or txt-file of the books
#
# Example use:
# python initializer.py generate --filename data/english_verbs.csv
# starts making 1-5 word palindromes from english_verbs.csv and process may take hours to days depending
# your computer's speed.
# Once all four needed files, such as new_nouns_palindromes.csv, new_verbs_palindromes.csv,
#     new_adjectives_palindromes.csv and new_long_text_palindromes.csv are generated, then you can execute
#     convert, which makes single palindromes.json- file for the game Machine Learning part

import argparse
import asyncio
import string
import logging
import os
import csv
import json
import requests
import pandas as pd
import textwrap
from nltk.corpus import wordnet as wn
import re

# Debug flags
debug_feeder = False
debug_downloader = True
debug_maker = True

# Global
begin_word = ""

# Logging setup
logger = logging.getLogger()
logger.setLevel(logging.ERROR)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
file_handler = logging.FileHandler('downloader_errors.log')
file_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Config loading
try:
    with open('data/runtimeconfig.json', 'r') as config_file:
        data = json.load(config_file)
        data_path = data.get('data_path')
        verbs_file = data_path + data.get('verbs_file')
        adjectives_file = data_path + data.get('adjectives_file')
        nouns_file = data_path + data.get('nouns_file')
        long_sentences_file = data_path + data.get('long_text_file')
        new_palindromes_file = data_path + data.get('new_palindromes_file')
        converted_palindromes_file = data_path + data.get('converted_palindromes_file')
        new_subs_palindromes_file = data_path + data.get('new_subs_palindromes_file')
        new_verb_palindromes_file = data_path + data.get('new_verb_palindromes_file')
        new_adj_palindromes_file = data_path + data.get('new_adj_palindromes_file')
        new_long_text_palindromes_file = data_path + data.get('new_long_text_palindromes_file')
except (OSError, json.JSONDecodeError) as err:
    logger.error(f"Error with runtimeconfig.json: {err}")
    print(f"Error with runtimeconfig.json: {err}")
    exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Download books or generate palindromes.")
    parser.add_argument(
        "action",
        choices=["download", "generate"],
        help="Action to perform: download or generate"
    )
    parser.add_argument(
        "--filename",
        help="File to use for palindrome generation (required for generate)",
        default=None
    )
    return parser.parse_args()

class Downloader(object):
    def __init__(self):
        self.books = {
            "Pride and Prejudice": "https://www.gutenberg.org/files/1342/1342-0.txt",
            "Moby Dick": "https://www.gutenberg.org/files/2701/2701-0.txt",
            "A Tale of Two Cities": "https://www.gutenberg.org/files/98/98-0.txt"
        }

    def download_book(self):
        merged_text = ""
        for title, url in self.books.items():
            try:
                if debug_feeder:
                    print(f"Downloading {title}...")
                response = requests.get(url)
                response.raise_for_status()
                text = response.text
                start_marker = "*** START OF"
                end_marker = "*** END OF"
                if start_marker in text:
                    text = text.split(start_marker, 1)[-1]
                if end_marker in text:
                    text = text.split(end_marker, 1)[0]
                merged_text += f"\n\n--- {title} ---\n\n{text.strip()}\n\n"

            except Exception as e:
                if debug_downloader :
                    print(f"Failed to download {title}: {e}")
                    logger.error(f"Failed to download {title}: {e}")
                pass

        # save to txt-file
        with open(long_sentences_file, 'w', encoding='utf-8') as f:
            f.write(merged_text)

        if debug_downloader :
            print(f"\nAll books saved to: {long_sentences_file}")

    @staticmethod
    def get_words(pos):
        words = set()
        for synset in wn.all_synsets(pos):
            for lemma in synset.lemmas():
                words.add(lemma.name().lower().replace('_', ''))
        return list(words)

    def download_v_a_n(self):
        verbs = self.get_words('v')
        adjectives = self.get_words('a')
        nouns = self.get_words('n')

        # Save to CSV
        with open(verbs_file, 'w', encoding='utf-8') as f:
            for v in verbs:
                f.write(f"{v}\n")

        with open(adjectives_file, 'w', encoding='utf-8') as f:
            for a in adjectives:
                f.write(f"{a}\n")

        with open(nouns_file, 'w', encoding='utf-8') as f:
            for n in nouns:
                f.write(f"{n}\n")

class FEEDER(object):
    # ANSI escape codes for colors
    COLOR_RESET = "\033[0m"
    COLOR_GREEN = "\033[92m"
    COLOR_YELLOW = "\033[93m"
    COLOR_BLUE = "\033[94m"
    COLOR_RED = "\033[91m"

    def __init__(self):
        """
               Constructor:
               Input (from runtimeconfig.json): verb, adjectives, nouns, long sentences (from book etc),
               and filename for new palindromes.

               Args: debug
        """
        self.sentences = None
        self.debug = debug_feeder
        self.clean_verbs = set()
        self.clean_adjectives = set()
        self.clean_nouns = set()
        self.extracted_words = set()
        self.verb_anagrams = set()
        self.adj_anagrams = set()
        self.subs_anagrams = set()
        self.long_anagrams = set()
        self.new_palindromes = []
        self.valid_start_letters = set()

        if verbs_file and os.path.exists(verbs_file):
            self.verbs = self.load_words(verbs_file)
            self.clean_verbs = set(self.remove_duplicates(self.verbs))  # remove duplicates
            self.word_anagrams_in_lists(self.clean_verbs, self.verb_anagrams)  # find anagramic words
            if self.debug:
                print(f"{self.COLOR_GREEN}Clean verbs loaded: {self.clean_verbs}{self.COLOR_RESET}")
                print(f"{self.COLOR_GREEN}Verb anagrams: {self.verb_anagrams}{self.COLOR_RESET}")

        if adjectives_file and os.path.exists(adjectives_file):
            self.adjectives = self.load_words(adjectives_file)
            self.clean_adjectives = set(self.remove_duplicates(self.adjectives))
            self.word_anagrams_in_lists(self.clean_adjectives, self.adj_anagrams)
            if self.debug:
                print(f"{self.COLOR_YELLOW}Clean adjectives loaded: {self.clean_adjectives}{self.COLOR_RESET}")
                print(f"{self.COLOR_YELLOW}Adjective anagrams: {self.adj_anagrams}{self.COLOR_RESET}")

        if nouns_file and os.path.exists(nouns_file):
            self.nouns = self.load_words(nouns_file)
            self.clean_nouns = set(self.remove_duplicates(self.nouns))
            self.word_anagrams_in_lists(self.clean_nouns, self.subs_anagrams)
            if self.debug:
                print(f"{self.COLOR_BLUE}Clean nouns loaded: {self.clean_nouns}{self.COLOR_RESET}")
                print(f"{self.COLOR_BLUE}Noun anagrams: {self.subs_anagrams}{self.COLOR_RESET}")

        if long_sentences_file and os.path.exists(long_sentences_file):
            self.long_sentences = self.load_sentences(long_sentences_file)  # note! txt-file!
            self.clean_long_sentences = set(self.remove_duplicates(self.long_sentences))
            self.extracted_words = self.remove_duplicates(self.extract_words_from_sentences(self.clean_long_sentences))
            filtered_words = [word for word in self.extracted_words if len(word) >= 2]  # filter words less than 2 chars
            self.extracted_words = filtered_words
            adjectives_set = set(self.clean_adjectives) if adjectives_file else set()
            verbs_set = set(self.clean_verbs) if verbs_file else set()
            nouns_set = set(self.clean_nouns) if nouns_file else set()
            self.extracted_words = [word for word in self.extracted_words
                                    if word not in adjectives_set and word not in verbs_set and word
                                    not in nouns_set]
            self.word_anagrams_in_lists(self.extracted_words, self.long_anagrams)
            if self.debug:
                print(f"{self.COLOR_RED}Words from long sentences after cleaning: "
                      f"{self.extracted_words}{self.COLOR_RESET}")

    def check_palindromes(self, word_list):
        """ Verify if palindrome = anagram too """
        return [word for word in word_list if word == word[::-1]]

    def load_words(self, file_name):
        """ Reads csv-file containing words, comma separated """
        try:
            with open(file_name, newline='') as f:
                reader = csv.reader(f)
                self.extracted_words = [row[0] for row in reader if row]
            return self.extracted_words
        except Exception as e:
            logger.error("Error: %s", e)
            if self.debug:
                print("Error: %s", e)

    def load_text_rows(self, file_name):
        """ Reads csv-file containing words, comma separated """
        try:
            with open(file_name, "r") as file:  # comma separated values
                text_rows = list(csv.reader(file, delimiter=","))
            text_rows = [item for sublist in text_rows for item in sublist if item]
            return text_rows
        except Exception as e:
            logger.error("Error: %s", e)
            if self.debug:
                print("Error: %s", e)

    def load_sentences(self, file_name):
        """ Reads txt-file containing lines """
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                self.sentences = f.readlines()
            self.sentences = [line.strip() for line in self.sentences if line.strip()]
            return self.sentences
        except Exception as e:
            logger.error("Error: %s", e)
            if self.debug:
                print("Error: %s", e)

    def extract_words_from_sentences(self, sentences):
        """ Strip words our from lines """
        words_out = []
        for sentence in sentences:
            words_in_sentence = sentence.split()
            cleaned_words = [self.clean_text(word) for word in words_in_sentence]
            words_out.extend(cleaned_words)
        return words_out

    @staticmethod
    def clean_text(text):
        """ Clean text, leave only alphabets """
        # for Finnish text = re.sub(r'[^a-zA-ZäöåÄÖÅ]', '', text)
        text = re.sub(r'[^a-zA-Z]', '', text)
        return text.lower()

    @staticmethod
    def remove_duplicates(input_list):
        """ Remove duplicate entries from lists """
        seen = set()
        unique_list = []
        for item in input_list:
            if item not in seen:
                unique_list.append(item)
                seen.add(item)
        return unique_list

    def remove_duplicates_with_spaces(self, palindrome_list):
        """ Remove duplicates from a list, preserving spaces and keeping the original order """
        seen = set()  # To track seen palindromes
        unique_palindromes = []  # List to store unique palindromes

        for palindrome in palindrome_list:
            cleaned_palindrome = palindrome.lower().strip()  # Clean spaces and make lowercase
            if cleaned_palindrome not in seen:
                unique_palindromes.append(palindrome)  # Add original (with spaces)
                seen.add(cleaned_palindrome)  # Track cleaned version to avoid duplicates

        return unique_palindromes

    def save_new_palindromes(self, palindromes, file_name):
        # Load existing palindromes from the file
        existing_palindromes = set()  # Using a set to avoid duplicates
        if os.path.exists(file_name):
            try:
                with open(file_name, 'r', encoding='utf-8') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        if row:
                            existing_palindromes.add(row[0].strip())  # Add existing palindromes to the set
            except Exception as e:
                logger.error("Error: %s", e)
                if self.debug:
                    print("Error: %s", e)

        # Remove duplicates in the new palindromes (and check against existing palindromes)
        new_unique_palindromes = self.remove_duplicates_with_spaces(palindromes)
        new_palindromes_to_save = [p for p in new_unique_palindromes if p.strip() not in existing_palindromes]

        # Append the new, unique palindromes to the file
        if new_palindromes_to_save:
            try:
                with open(file_name, 'a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    for palindrome in new_palindromes_to_save:
                        writer.writerow([palindrome])
            except Exception as e:
                logger.error("Error: %s", e)
                if self.debug:
                    print("Error: %s", e)
            if self.debug:
                print(f"Saved {len(new_palindromes_to_save)} new palindromes to {file_name}")

    def word_anagrams_in_lists(self, word_list, anagram_list):
        """ Generalized palindrome check for word lists """
        for word in word_list:
            if word == word[::-1]:
                anagram_list.add(word)
                if self.debug:
                    print(f"Anagram found: {word}")


class PalindromeMaker:
    def __init__(self, feeder, max_palindromes=20000):
        self.feeder = feeder
        self.debug = debug_maker
        self.max_palindromes = max_palindromes
        self.status = "Not running"
        self.chosen_wordlist = None
        self.new_file = None
        self.cancel_requested = False
        self.fail_counter = 0
        self.found_counter = 0

    def is_anagram(self, text):
        """ Check if anagram (mirror) """
        return text == text[::-1]

    def make_symmetric(self, text):
        """ Make symmetric = mirror"""
        return text + text[::-1]

    async def iterate_alphabet_characters(self, word, position):
        found_palindrome = False
        first_letter = ""
        palindrome = ""
        # Finnish: alphabet = string.ascii_lowercase + 'äöå'
        alphabet = string.ascii_lowercase
        index = len(word) + position
        for letter in alphabet:
            if self.cancel_requested:
                if self.debug:
                    print("Generation cancelled!", flush=True)
                return False
            new_word = word[:index] + letter + word[index:] + word[::-1]
            if self.make_sense(new_word):
                if new_word == new_word[::-1]:
                    found_palindrome = True
                    first_letter = letter
                    palindrome = new_word
            await asyncio.sleep(0)
        if found_palindrome:
            self.feeder.new_palindromes.append(palindrome)
            used_words = set()
            await self.extend_palindrome_second_phase(palindrome, first_letter, index, used_words)
            return True
        else:
            self.fail_counter += 1
            return False

    def find_palindrome_extensions_first_letter(self, first_letter, used_words):
        """ Return verbs etc. based on first letter of the word or sentence """
        ext_verbs = [w for w in self.feeder.clean_verbs if w.lower().startswith(first_letter) and w not in used_words]
        ext_adjectives = [w for w in self.feeder.clean_adjectives if
                          w.lower().startswith(first_letter) and w not in used_words]
        ext_nouns = [w for w in self.feeder.clean_nouns if
                            w.lower().startswith(first_letter) and w not in used_words]
        ext_txt_words = [w for w in self.feeder.extracted_words if w.lower().startswith(first_letter) and w not in used_words]
        return ext_verbs + ext_adjectives + ext_nouns + ext_txt_words

    async def extend_palindrome_second_phase(self, palindrome, first_letter, index, used_words):
        extensions = self.find_palindrome_extensions_first_letter(first_letter, used_words)
        for ext_word in extensions:
            extended_palindrome = ' '.join([palindrome[:index], ext_word, palindrome[index:]])
            if extended_palindrome.replace(' ', '') == extended_palindrome.replace(' ', '')[::-1]:
                self.feeder.new_palindromes.append(extended_palindrome)
                used_words.add(ext_word)
                await self.extend_palindrome_second_phase(
                    extended_palindrome.replace(' ', ''), first_letter, index + len(ext_word), used_words
                )
            await asyncio.sleep(0)

    def make_sense(self, anagram):
        """ Test if word makes sense = is found from vocabulary based on FEEDER words """
        first_letter = anagram[0].lower()
        first_match = False

        # Check first part
        matching_verbs = [w for w in self.feeder.clean_verbs if w.lower().startswith(first_letter)]
        matching_adjectives = [w for w in self.feeder.clean_adjectives if w.lower().startswith(first_letter)]
        matching_nouns = [w for w in self.feeder.clean_nouns if w.lower().startswith(first_letter)]
        matching_txt_words = [w for w in self.feeder.extracted_words if w.lower().startswith(first_letter)]

        # Check if word is in the lists
        if (begin_word in matching_verbs or begin_word in matching_adjectives or begin_word in matching_nouns
                or begin_word in matching_txt_words):
            first_match = True

        # If the word is found, continue to next word
        if first_match:
            remaining_part = anagram[len(begin_word):]  # rest part
            if self.check_remaining_part_second_phase(remaining_part) is True:
                return True
        else:
            return False

    def convert_new_csv_to_json(self):
        # Files which must exist
        required_files = [
            new_subs_palindromes_file,
            new_verb_palindromes_file,
            new_adj_palindromes_file,
            new_long_text_palindromes_file
        ]

        # Check existence
        missing_files = [file for file in required_files if not os.path.exists(file)]

        if missing_files:
            self.status = f"Missing files: {', '.join(missing_files)}. Need all four before proceeding!"
            if self.debug:
                print(f"Missing files: {', '.join(missing_files)}. Need all four before proceeding!")
            return False

        # Keep track of counts for debugging
        counts = {}
        dataframes = []

        # Load csv files prior to conversion
        for file_path in required_files:
            try:
                df = pd.read_csv(file_path, header=None)
                # Store original count
                counts[file_path] = len(df)
                if len(df) > 0:
                    dataframes.append(df)
                else:
                    if self.debug:
                        print(f"Warning: {file_path} is empty")
            except Exception as e:
                self.status = f"Error reading {file_path}: {e}"
                logger.error(f"Error reading {file_path}: {e}")
                if self.debug:
                    print(f"Error: {e}")
                return False

        # Check that at least one file has data
        if not dataframes:
            self.status = "All files are empty or failed to load."
            if self.debug:
                print("All files are empty or failed to load.")
            return False

        # Combine dataframes
        combined_palindromes = pd.concat(dataframes, ignore_index=True)
        total_before = len(combined_palindromes)

        # Remove duplicates
        combined_palindromes.drop_duplicates(subset=[0], inplace=True)
        total_after = len(combined_palindromes)

        # Get list of palindromes
        palindromes_list = combined_palindromes[0].tolist()

        # Print counts for debugging
        if self.debug:
            print(f"Original counts: {counts}")
            print(f"Before deduplication: {total_before}, After: {total_after}")
            print(f"Duplicates removed: {total_before - total_after}")

        # Save to json
        try:
            with open(converted_palindromes_file, 'w', encoding='utf-8') as f:
                json.dump(palindromes_list, f, ensure_ascii=False, indent=4)
            # If successful, set status message and return True
            self.status = f"CSV to JSON conversion complete and saved! Total palindromes: {len(palindromes_list)}"
            if self.debug:
                print(self.status)
            return True
        except Exception as e:
            self.status = f"Error saving to {converted_palindromes_file}: {e}"
            logger.error(f"Error saving to {converted_palindromes_file}: {e}")
            if self.debug:
                print(f"Error: {e}")
            return False

    def check_remaining_part_second_phase(self, remaining_part):
        """ If there is more to check about the palindrome"""
        if remaining_part:
            first_letter = remaining_part[0].lower()

            # Check rest part in vocabulary
            matching_verbs = [w for w in self.feeder.clean_verbs if w.lower().startswith(first_letter)]
            matching_adjectives = [w for w in self.feeder.clean_adjectives if w.lower().startswith(first_letter)]
            matching_nouns = [w for w in self.feeder.clean_nouns if w.lower().startswith(first_letter)]
            matching_txt_words = [w for w in self.feeder.extracted_words if w.lower().startswith(first_letter)]

            # If found, return true
            if (remaining_part in matching_verbs or remaining_part in matching_adjectives
                    or remaining_part in matching_nouns or remaining_part in matching_txt_words):
                return True

    def cancel_generation(self):
        """ Set interrupt handler """
        self.cancel_requested = True

    def save_progress(self):
        """ Save the current progress of new palindromes - control from GENERATOR class!"""
        self.feeder.save_new_palindromes(self.feeder.new_palindromes, self.new_file)
        self.status = "Progress saved!"
        if self.debug:
            print(f"Progress saved!")

    async def make_palindromes_for_learning(self, max_palindromes):
        global begin_word
        iteration_count = 0
        if self.chosen_wordlist and not self.cancel_requested:
            for begin_word in self.chosen_wordlist:
                if self.cancel_requested:
                    if self.debug:
                        print("Generation interrupted!", flush=True)
                    return
                if iteration_count >= max_palindromes:
                    self.status = f"Reached maximum palindromes: {max_palindromes}"
                    print(self.status, flush=True)
                    break
                if await self.iterate_alphabet_characters(begin_word, 0):
                    iteration_count += 1
                await asyncio.sleep(0)
            if not self.cancel_requested:
                self.save_progress()
        else:
            print("Wordlist empty or generation cancelled.", flush=True)

    async def next_level_iterator(self):
        """ If first and/or second level palindromes are found, tries to add more text in between """
        previous_length = len(self.feeder.new_palindromes)

        while True:
            # Check if the palindrome list has grown
            if len(self.feeder.new_palindromes) > previous_length:
                self.status = "New palindrome added! Adding anagram word to both ends..."
                if self.debug:
                    print("New palindrome added! Adding anagram word to both ends...")
                new_palindrome = self.feeder.new_palindromes[-1]  # Get the latest palindrome

                # Extend the palindrome with anagrams
                await self.extend_palindrome_next_level(new_palindrome)

                # Update the previous length
                previous_length = len(self.feeder.new_palindromes)

            await asyncio.sleep(1)  # Sleep to avoid constant checking

    async def extend_palindrome_next_level(self, palindrome):
        """Try to extend the palindrome using anagram words"""
        anagram_sources = [
            self.feeder.subs_anagrams,
            self.feeder.verb_anagrams,
            self.feeder.adj_anagrams,
            self.feeder.long_anagrams
        ]

        # For each source of anagrams, try to extend the palindrome
        for source in anagram_sources:
            for word in source:
                # Create new palindromes by adding the word at both ends
                extended_start = word + " " + palindrome + " " + word[::-1]  # Add at both ends
                self.feeder.new_palindromes.append(extended_start)

    def format_list(self, data_list, width=80):
        """ Format list to display with a specified width per row """
        # Join the list into a single string and wrap it to the specified width
        return '\n'.join(textwrap.wrap(', '.join(data_list), width))

    async def print_status(self):
        while not self.cancel_requested:
            self.found_counter = len(self.feeder.new_palindromes)
            latest_palindrome = self.feeder.new_palindromes[-1] if self.feeder.new_palindromes else "None"
            # Truncate to fit terminal (e.g., 30 chars for palindrome)
            latest_palindrome = (latest_palindrome[:27] + "...") if len(latest_palindrome) > 27 else latest_palindrome
            status_message = (
                f"\rTries: {self.fail_counter}  Found: {self.found_counter}  "
                f"Current: {begin_word[:15]:<15}  Latest: {latest_palindrome:<30}"
            )
            print(status_message, end='', flush=True)
            await asyncio.sleep(1)
        print("\r" + " " * 80, end='\r', flush=True)
        print("Status updates stopped.", flush=True)


class CLIHandler:
    def __init__(self):
        self.downloader = Downloader()
        self.feeder = FEEDER()
        self.maker = PalindromeMaker(feeder=self.feeder)

    async def handle_download(self):
        print("Starting download...", flush=True)
        self.downloader.download_book()
        self.downloader.download_v_a_n()
        print("Download complete.", flush=True)

    async def handle_generate(self, filename):
        if not filename:
            print("Error: --filename is required for generate action.", flush=True)
            print("Filenames: %s or %s or %s or %s " % (verbs_file, adjectives_file, nouns_file, long_sentences_file))

            return
        print(f"Generating palindromes using {filename}...", flush=True)
        self.feeder = FEEDER()
        if filename.endswith(".csv") or filename.endswith(".txt"):
            if os.path.exists(filename):
                if filename == verbs_file:
                    self.maker.chosen_wordlist = list(self.feeder.clean_verbs)
                    self.maker.new_file = new_verb_palindromes_file
                elif filename == adjectives_file:
                    self.maker.chosen_wordlist = list(self.feeder.clean_adjectives)
                    self.maker.new_file = new_adj_palindromes_file
                elif filename == nouns_file:
                    self.maker.chosen_wordlist = list(self.feeder.clean_nouns)
                    self.maker.new_file = new_subs_palindromes_file
                elif filename == long_sentences_file:
                    self.maker.chosen_wordlist = list(self.feeder.extracted_words)
                    self.maker.new_file = new_long_text_palindromes_file
                else:
                    print(f"Error: Unsupported file {filename}", flush=True)
                    return
            else:
                print(f"Error: File {filename} does not exist.", flush=True)
                return
        else:
            print("Error: Filename must be a .csv or .txt file.", flush=True)
            return

        loop = asyncio.get_event_loop()
        status_task = loop.create_task(self.maker.print_status())
        await self.maker.make_palindromes_for_learning(self.maker.max_palindromes)
        status_task.cancel()
        print("Generation complete.", flush=True)

    async def handle_convert(self):
        print("Starting CSV to JSON conversion...", flush=True)
        success = self.maker.convert_new_csv_to_json()
        if success:
            print(f"Success: {self.maker.status}, now you can copy this file to the game root directory.", flush=True)
        else:
            print(f"Error: {self.maker.status}", flush=True)

    async def run(self):
        args = parse_args()
        if args.action == "download":
            await self.handle_download()
        elif args.action == "generate":
            await self.handle_generate(args.filename)
        elif args.action == "convert":
            await self.handle_convert()

def parse_args():
    parser = argparse.ArgumentParser(description="Download books or generate palindromes.")
    parser.add_argument(
        "action",
        choices=["download", "generate", "convert"],
        help="Action to perform: download, generate, or convert"
    )
    parser.add_argument(
        "--filename",
        help="File to use for palindrome generation (required for generate)",
        default=None
    )
    return parser.parse_args()

async def main():
    handler = CLIHandler()
    await handler.run()

if __name__ == "__main__":
    asyncio.run(main())