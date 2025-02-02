import json
import sys

class WordPieceTokenizer:
    def __init__(self):
        self.vocab = []

    def preprocess_data(self, text):
        # Convert text to lowercase
        preprocessed_text = text.lower()

        # Remove punctuations, special characters and numbers
        preprocessed_text = " ".join("".join(c if c.isalpha() else ' ' for c in word) for word in preprocessed_text.split())

        # Remove extra whitespaces
        preprocessed_text = " ".join(preprocessed_text.split())

        return preprocessed_text
    
    def construct_vocabulary(self, corpus, vocabulary_size):
        # Split the corpus into list of words
        tokenized_corpus = [sentence.split() for sentence in corpus]

        # Initialize word frequency and split corpus into initial tokens
        word_freq = {}
        split_corpus = {}
        for word_array in tokenized_corpus:
            for word in word_array:
                word_freq[word] = word_freq.get(word, 0) + 1
                if word not in split_corpus:
                    # Subword Splitting
                    split_corpus[word] = [word[0]] + ['##' + char for char in word[1:]]

        # Initialize vocabulary with unique character-based tokens
        vocabulary = list(set([char for _, split in split_corpus.items() for char in split]))
        vocabulary.sort()
        vocabulary = ["[UNK]", "[PAD]"] + vocabulary  # Add special tokens

        # Merge tokens iteratively until vocabulary size is reached
        while len(vocabulary) <= vocabulary_size:
            pair_freq = {}  # Frequency of token pairs
            token_freq = {}  # Frequency of individual tokens

            # Compute token frequencies
            for word, split in split_corpus.items():
                for token in split:
                    token_freq[token] = token_freq.get(token, 0) + word_freq[word]

            # Compute pair frequencies
            for word, split in split_corpus.items():
                for idx in range(len(split) - 1):
                    pair = (split[idx], split[idx + 1])
                    pair_freq[pair] = pair_freq.get(pair, 0) + word_freq[word]

            # Stop if no more pairs exist
            if not pair_freq:
                break

            # Select the most frequent token pair to merge and add it to the vocabulary
            max_pair = max(pair_freq, key=lambda pair: self.__score(pair_freq, token_freq, pair))
            max_token = max_pair[0] + max_pair[1][2:]
            vocabulary.append(max_token)

            # Update split_corpus by replacing merged pairs with new token
            for word, split in split_corpus.items():
                updated_split = []
                idx = 0
                while idx < len(split):
                    if idx + 1 < len(split) and split[idx] == max_pair[0] and split[idx + 1] == max_pair[1]:
                        updated_split.append(max_token)
                        idx += 2
                    else:
                        updated_split.append(split[idx])
                        idx += 1
                split_corpus[word] = updated_split

        self.vocab = vocabulary[:vocabulary_size]
        file_path = "vocabulary.txt"

        # Save vocabulary to file
        with open(file_path, "w", encoding="utf-8") as f:
            for token in self.vocab:
                f.write(token + "\n")
        print(f"Vocabulary has been constructed in vocabulary.txt")
    
    def tokenize(self, text):
        tokens = []
        word_array = text.split()

        # Traverse the sentence to tokenize each word
        for word in word_array:
            start_idx = 0
            while start_idx < len(word):
                # Prefix subwords with the delimiter except the starting subword of a word
                marker = '##' if start_idx != 0 else ''
                end_idx = len(word) - 1
                unk_flag = True

                # Find the longest matching subword in vocabulary
                while end_idx >= start_idx:
                    if marker + word[start_idx:end_idx + 1] in self.vocab:
                        tokens.append(marker + word[start_idx:end_idx + 1])
                        start_idx = end_idx + 1
                        unk_flag = False
                        break
                    else:
                        end_idx -= 1

                # If no subword match is found, assign [UNK] token
                if unk_flag:
                    tokens.append('[UNK]')
                    start_idx += 1

        return tokens

    # Private function to compute the score of pair of tokens for merging
    def __score(self, pair_freq, token_freq, pair):
        (token1, token2) = pair
        return pair_freq[pair] / (token_freq[token1] * token_freq[token2])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python word_piece_tokenizer.py <json_file>")
        sys.exit(1)

    try:
        corpus = []
        corpus_file = 'corpus.txt'

        # Load corpus from file
        with open(corpus_file, 'r', encoding='utf-8') as file:
            for line in file:
                corpus.append(line.strip())

        try:
            json_file = sys.argv[1]
            
            # Load input JSON file
            with open(json_file, "r", encoding="utf-8") as file:
                data = json.load(file)

            tokenizer = WordPieceTokenizer()
            corpus = [tokenizer.preprocess_data(line) for line in corpus]  # Preprocess corpus
            tokenizer.construct_vocabulary(corpus, vocabulary_size=1000)  # Construct vocabulary
            
            tokenized_data = {}

            # Tokenize each sentence in input JSON
            for item in data:
                sentence = tokenizer.preprocess_data(item['sentence'])
                tokens = tokenizer.tokenize(sentence)
                tokenized_data[int(item['id'])] = tokens

            # Save tokenized output to JSON file
            with open("tokenized.json", "w", encoding="utf-8") as file:
                json.dump(tokenized_data, file, ensure_ascii=False, indent=4)
            print(f"Data has been tokenized in tokenized.json")

        except FileNotFoundError:
            print(f"Error: File '{json_file}' not found.")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in '{json_file}'.")
    
    except FileNotFoundError:
        print(f"Error: File {corpus_file} not found.")
