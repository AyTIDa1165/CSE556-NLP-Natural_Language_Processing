# CSE556-NLP-Natural_Language_Processing
This repository contains the full codebase, reports, and results for **three major NLP assignments** submitted for the CSE 556 course on Natural Language Processing. Each assignment features a structured progression, beginning with foundational techniques and then building upon them as students advance through each of its three tasks. The code is implemented using Python, PyTorch, HuggingFace Transformers, and relevant libraries as per task requirements.

## Weights
The model weights for all assignments and tasks can be found at the following link:
https://drive.google.com/drive/folders/1LS5sQhi_YZXnAZAPn9XFGYvITss3Otau?usp=sharing


## 🔍 Assignment Highlights & Learning Outcomes

### ✅ **Assignment 1: Foundational NLP Modeling**

1. **WordPiece Tokenizer**

   * Built from scratch using only standard Python libraries.
   * Learned core principles of subword tokenization, vocabulary construction, and preprocessing.

2. **Word2Vec (CBOW)**

   * Implemented using custom `Dataset` and `Model` classes in PyTorch.
   * Learned vector space semantics, cosine similarity, and embedding quality evaluation.

3. **Neural Language Model (MLP)**

   * Developed three variations of MLP architectures for next-token prediction.
   * Integrated previous tokenizer and embeddings, explored design decisions impacting perplexity and accuracy.

### ✅ **Assignment 2: Aspect-Based Sentiment and QA**

1. **Aspect Term Extraction (ATE)**

   * Trained RNN/GRU models using BIO encoding and pretrained embeddings.
   * Focused on sequence labeling and handling multi-aspect annotations.

2. **Aspect-Based Sentiment Analysis (ABSA)**

   * Designed deep models for ABSA using FastText, GloVe, and transformer-based embeddings.
   * Compared traditional RNNs with fine-tuned BERT, BART, and RoBERTa models.

3. **SpanBERT QA (SQuAD v2)**

   * Fine-tuned both SpanBERT and SpanBERT+CRF for extractive question answering.
   * Understood answer span extraction, CRF integration, and exact-match evaluation.

### ✅ **Assignment 3: Advanced Generation and Multimodal Modeling**

1. **Transformer from Scratch**

   * Reproduced core components (attention, masking, positional encodings, etc.) from the ground up.
   * Gained in-depth intuition of architecture and training via Shakespearean language modeling.

2. **Claim Normalization (CLAN Dataset)**

   * Fine-tuned BART and T5 for controlled generation from noisy, real-world social media posts.
   * Practiced performance tuning, text preprocessing, and evaluation using ROUGE, BLEU, and BERTScore.

3. **Multimodal Sarcasm Explanation (MuSE)**

   * Implemented the TURBO model using ViT and BART.
   * Built shared fusion mechanisms to blend textual and visual information.
   * Generated natural language explanations and evaluated with multiple generation metrics.

---

## 🧠 Skills Demonstrated

* **From-scratch implementations** (tokenizers, embeddings, attention)
* **Model fine-tuning** (BERT, RoBERTa, BART, SpanBERT, T5)
* **Sequence labeling, QA, sentiment analysis, claim generation**
* **Multimodal learning (image + text)**
* **Metric-based evaluation** (Perplexity, F1, Accuracy, ROUGE, BLEU, BERTScore, EM)
* **Model performance analysis and architecture experimentation**


## 📌 Notes

* Each assignment folder includes:

  * Complete code for all three tasks
  * Saved models and results
  * A detailed PDF report summarizing architecture, hyperparameters, plots, evaluation, and insights.

---

For any queries, feel free to reach out the authors!

📫 \[[Aarya Gupta](mailto:aarya22006@iiitd.ac.in)], \[[Aditya Aggarwal](mailto:aditya22028@iiitd.ac.in)], \[[Arpan Verma](mailto:arpan22105@iiitd.ac.in)], 
