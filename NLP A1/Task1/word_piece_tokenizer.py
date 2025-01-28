import pandas as pd
import json
import sys

class WordPieceTokenizer:
    def __init__(self):
        self.vocab = []

    def preprocess_data(self, text):
        preprocessed_text = text.lower()
        preprocessed_text = " ".join("".join(c if c.isalpha() else ' ' for c in word) for word in preprocessed_text.split())
        preprocessed_text = " ".join(preprocessed_text.split())
        return preprocessed_text
    
    def construct_vocabulary(self, corpus, vocabulary_size):
        tokenized_corpus = [sentence.split() for sentence in corpus]
        unique_words = list(set([word for array in tokenized_corpus for word in array]))
        split_corpus = [[word[0]] + ['##' + char for char in word[1:]] for word in unique_words]
        vocabulary = list(set([char for split in split_corpus for char in split]))
        
        for _ in range(vocabulary_size):
            pair_freq = {}
            token_freq = {}
            for split in split_corpus:
                for token in split:
                    token_freq[token] = token_freq.get(token, 0) + 1
                    token_freq[token] += 1

            for split in split_corpus:
                for idx in range(len(split) - 1):
                    pair = (split[idx], split[idx+1])
                    pair_freq[pair] = pair_freq.get(pair, 0) + 1
                    pair_freq[pair] += 1

            if(len(pair_freq) == 0):
                break
            
            max_pair = max(pair_freq, key=lambda pair: self.__score(pair_freq, token_freq, pair))
            max_token = max_pair[0] + max_pair[1][2:]
            vocabulary.append(max_token)

            updated_split_corpus = []
            for split in split_corpus:
                updated_split = []
                idx = 0
                while idx < len(split):
                    if idx+1 < len(split) and split[idx] == max_pair[0] and split[idx+1] == max_pair[1]:
                        updated_split.append(max_token)
                        idx += 2
                    else:
                        updated_split.append(split[idx])
                        idx += 1
                updated_split_corpus.append(updated_split)
            split_corpus = updated_split_corpus

        self.vocab = vocabulary
        file_path = "vocabulary.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            for token in vocabulary:
                f.write(token + "\n")
        print(f"Vocabulary has been constructed in vocabulary.txt")
    
    def tokenize(self, text):
        tokens = []
        word_array = text.split()
        for word in word_array:
            start_idx = 0
            while start_idx < len(word):
                marker = '##' if start_idx != 0 else ''
                end_idx = len(word)-1
                error_flag = True
                while end_idx >= start_idx:
                    if marker + word[start_idx:end_idx+1] in self.vocab:
                        tokens.append(marker + word[start_idx:end_idx+1])
                        start_idx = end_idx+1
                        error_flag = False
                        break
                    else:
                        end_idx -= 1
                if error_flag:
                    raise Exception(f"ERROR: Encountered unknown character while tokenizing {word}")
        return tokens

    def __score(self, pair_freq, token_freq, pair):
        (token1, token2) = pair
        return (pair_freq[pair] / (token_freq[token1] * token_freq[token2]))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python word_piece_tokenizer.py <json_file>")
        sys.exit(1)

    try:
        df = pd.read_csv("corpus.txt", header=None, names=["text"])

        try:
            json_file = sys.argv[1]
            with open(json_file, "r", encoding="utf-8") as file:
                data = json.load(file)

            tokenizer = WordPieceTokenizer()
            corpus = df['text'].tolist()
            corpus= [tokenizer.preprocess_data(line) for line in corpus]
            tokenizer.construct_vocabulary(corpus, vocabulary_size=1000)
            
            tokenized_data = {}

            for item in data:
                sentence = tokenizer.preprocess_data(item['sentence'])
                tokens = tokenizer.tokenize(sentence)
                tokenized_data[int(item['id'])] = tokens

            with open("tokenized.json", "w", encoding="utf-8") as file:
                json.dump(tokenized_data, file, ensure_ascii=False, indent=4)
            print(f"Data has been tokenized in tokenized.json")

        except FileNotFoundError:
            print(f"Error: File '{json_file}' not found.")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in '{json_file}'.")
    except FileNotFoundError:
        print(f"Error: File corpus.txt not found.")