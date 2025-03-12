# import torch
# from task1 import WordPieceTokenizer
# from Task3 import NeuralLM3, predict_next_tokens  # Import your current model class

# # Configuration
# TEST_FILE = "test.txt"
# CONTEXT_SIZE = 3
# DEVICE = torch.device("cpu")
# MODEL_PATH = "neural_lm3.pth"  # Checkpoint saved with different key names

# # Load the tokenizer and vocabulary
# tokenizer = WordPieceTokenizer()
# with open("vocabulary.txt", "r", encoding="utf-8") as vf:
#     tokenizer.vocab = [line.strip() for line in vf]
# vocab_size = len(tokenizer.vocab)
# embed_dim = 256  # Make sure this matches the saved model's embedding dimension

# # Initialize your current model
# model = NeuralLM3(vocab_size, embed_dim, CONTEXT_SIZE, hidden_dim=256, dropout_rate=0.5).to(DEVICE)

# # Load the saved state dict
# state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

# # Remap keys: change 'embedding' -> 'embed', 'fc1' -> 'l1', 'fc2' -> 'l2', 'fc3' -> 'l3'
# new_state_dict = {}
# for key, value in state_dict.items():
#     new_key = key
#     if key.startswith("embedding"):
#         new_key = key.replace("embedding", "embed")
#     if key.startswith("fc1"):
#         new_key = new_key.replace("fc1", "l1")
#     if key.startswith("fc2"):
#         new_key = new_key.replace("fc2", "l2")
#     if key.startswith("fc3"):
#         new_key = new_key.replace("fc3", "l3")
#     # Optionally, if your current model expects 'out_layer' but the saved key is different,
#     # add a similar replacement here.
#     new_state_dict[new_key] = value

# # Load the remapped state dict into your model
# model.load_state_dict(new_state_dict)
# model.eval()

# # Run predictions on the test file
# with open(TEST_FILE, "r") as testf:
#     for line in testf:
#         sentence = line.strip()
#         if not sentence:
#             continue
#         predicted_tokens = predict_next_tokens(model, sentence, tokenizer, CONTEXT_SIZE, num_tokens=3, device=DEVICE)
#         print(f"\nInput Sentence: {sentence}")
#         print(f"Predicted Tokens: {predicted_tokens}")