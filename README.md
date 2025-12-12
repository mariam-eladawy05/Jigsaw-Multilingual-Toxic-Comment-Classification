# **Jigsaw Multilingual Toxic Comment Classification**

## *Overview*

This project builds a **multilingual toxic comment classifier** using three models: **Hybrid BiLSTM-GRU**, **RNN**, and **pretrained BERT**. It leverages **FastText embeddings** for multiple languages and handles heavy class imbalance in toxic comment detection.

The system is designed to:

* Detect toxic content across six toxicity categories.
* Support multiple languages including English, French, Dutch, Italian, Afrikaans, and Finnish.
* Compare training and validation performance across models with visual plots.

---

## *Dataset*

The project uses the **Jigsaw Multilingual Toxic Comment dataset**:

* **Samples:** 223,549 comments
* **Labels:** toxic, severe_toxic, obscene, threat, insult, identity_hate
* **Languages:** Multiple; English dominates (~210k comments)
* **Class distribution:** Highly imbalanced (e.g., severe_toxic 0.87%, threat 0.3%)

Data cleaning includes:

* Removing URLs, mentions, HTML tags, punctuation, digits
* Tokenization, lemmatization, stopword removal
* Padding sequences for model input

---

## *Features*

* Preprocessing pipeline for multilingual text
* FastText embeddings integration for six languages
* Models:

  * **Hybrid BiLSTM-GRU**
  * **Bidirectional RNN**
  * **Pretrained Multilingual BERT**
* Class weighting to handle imbalance
* Visual comparisons of training and validation accuracy and loss

---

## *Installation*

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

---

## *Usage*

1. Load the dataset (place in `data/` folder):

```python
train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')
```

2. Preprocess the text:

```python
train['clean_text'] = train['comment_text'].apply(clean_text)
```

3. Build embedding matrix with FastText:

```python
embedding_index = load_fasttext_bin(['en','fr','nl','it','af','fi'])
```

4. Train models:

```python
history_hybrid = hybrid_model.fit(X_train, y_train, validation_data=(X_val, y_val), ...)
history_rnn    = rnn_model.fit(X_train, y_train, validation_data=(X_val, y_val), ...)
```

5. Train BERT:

```python
for epoch in range(EPOCHS):
    train_acc, train_loss = train_epoch(bert, train_loader, optimizer, scheduler, device)
```

6. Visualize results:

```python
plt.plot(hybrid_train_acc, label='Hybrid Train Acc')
plt.plot(rnn_val_acc, label='RNN Val Acc')
plt.plot(bert_val_acc_list, label='BERT Val Acc')
```

---

## *Requirements*

See [requirements.txt](./requirements.txt) for full dependencies.

* Python 3.8+
* TensorFlow 2.x
* PyTorch
* Transformers
* Scikit-learn
* Gensim
* NLTK
* Matplotlib, Seaborn
* Langdetect
* Pandas, NumPy

---

## *Results*

The project provides plots for:

* Training vs validation accuracy
* Training vs validation loss
* Comparative performance across all three models

---
