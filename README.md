# 📝 Deep Learning in NLP

A hands-on collection of **Deep Learning and Natural Language Processing (NLP) projects** implemented with Python.

This repository explores the application of Deep Learning to **text and natural language data**, starting from fundamental text classification tasks and progressing toward more advanced NLP applications such as **Spam Detection, Hate Speech Detection, Sentiment Analysis, and Next Sentence Prediction using BERT**.

The projects are implemented as Jupyter Notebooks and combine theoretical concepts with practical implementations and experiments using modern Deep Learning frameworks.

---

## 📚 Contents

* [Projects](#-projects)

  * [SMS Spam Detection](#1-sms-spam-detection)
  * [Hate Speech Detection](#2-hate-speech-detection)
  * [Sentiment Analysis with RNN](#3-sentiment-analysis-with-rnn)
  * [Next Sentence Prediction using BERT](#4-next-sentence-prediction-using-bert)
* [NLP Tasks Covered](#-nlp-tasks-covered)
* [Repository Structure](#-repository-structure)
* [Datasets](#-datasets)
* [Technologies](#-technologies)
* [Installation](#-installation)
* [Usage](#-usage)
* [Learning Path](#-learning-path)
* [Future Improvements](#-future-improvements)
* [Learning Objectives](#-learning-objectives)
* [Contributing](#-contributing)
* [Author](#-author)

---

# 🚀 Projects

## 1. SMS Spam Detection

**Notebook:** `SMS_Spam_Detection.ipynb`

A practical **text classification** project that uses Deep Learning to classify SMS messages as **spam or legitimate (ham)**.

The project demonstrates how raw text can be transformed into numerical representations and used as input to a Neural Network.

### Main Concepts

* Natural Language Processing
* Text preprocessing
* Text cleaning
* Tokenization
* Vocabulary
* Text vectorization
* Word embeddings
* Binary text classification
* Neural Networks
* TensorFlow / Keras
* Model evaluation

### General Pipeline

```text
Raw SMS
   ↓
Text Preprocessing
   ↓
Tokenization
   ↓
Text Vectorization
   ↓
Word Embedding
   ↓
Neural Network
   ↓
Binary Classification
   ↓
Spam / Ham
```

This project provides a simple introduction to applying Deep Learning techniques to real-world text classification problems.

---

# 2. Hate Speech Detection

**Notebook:** `Hate_Speech_Detection.ipynb`

A Natural Language Processing project focused on detecting **hate speech in textual data** using Deep Learning.

The model learns linguistic patterns from labeled text and predicts whether a given text belongs to the target hate-speech category.

## Main Concepts

* NLP
* Text preprocessing
* Text normalization
* Tokenization
* Text vectorization
* Word embeddings
* Text classification
* Deep Neural Networks
* Binary / multi-class classification
* Model evaluation

### General Pipeline

```text
Input Text
     ↓
Text Preprocessing
     ↓
Tokenization
     ↓
Vectorization
     ↓
Embedding
     ↓
Deep Learning Model
     ↓
Classification
     ↓
Predicted Category
```

This project demonstrates how Deep Learning can be applied to automatically analyze large amounts of textual content.

---

# 3. Sentiment Analysis with RNN

**Notebook:** `Sentiment_Analysis.ipynb`

A sentiment analysis project that uses **Recurrent Neural Networks (RNNs)** to classify the sentiment expressed in textual data.

RNN-based architectures are designed to process sequential data and are particularly useful for understanding the order and context of words within a sentence.

## Main Concepts

* Natural Language Processing
* Sequential data
* Text preprocessing
* Tokenization
* Text vectorization
* Word embeddings
* Recurrent Neural Networks
* Sequence modeling
* Sentiment classification
* Model evaluation

### General Pipeline

```text
Input Text
     ↓
Text Preprocessing
     ↓
Tokenization
     ↓
Text Vectorization
     ↓
Embedding
     ↓
RNN
     ↓
Dense Layer
     ↓
Sentiment Prediction
```

The project demonstrates how RNNs can learn sequential patterns in natural language and use them for sentiment classification.

---

# 4. Next Sentence Prediction using BERT

**Notebook:** `Next_Sentence_Prediction.ipynb`

This project explores **Next Sentence Prediction (NSP)** using **BERT (Bidirectional Encoder Representations from Transformers)**.

Unlike traditional RNN-based approaches, BERT is based on the **Transformer architecture** and uses bidirectional contextual representations to understand relationships between words and sentences.

Next Sentence Prediction is a pretraining task originally introduced with BERT to determine whether one sentence logically follows another.

## Main Concepts

* Transformers
* BERT
* Attention Mechanism
* Self-Attention
* Bidirectional Context
* Sentence Pair Classification
* Next Sentence Prediction
* Tokenization
* BERT Tokenizer
* Pretrained Language Models
* Transfer Learning

### General Pipeline

```text
Sentence A
     +
Sentence B
     ↓
BERT Tokenizer
     ↓
Token IDs
     ↓
BERT
     ↓
Transformer Encoder
     ↓
Contextual Representations
     ↓
Classification Layer
     ↓
Next Sentence Prediction
```

The model determines whether the second sentence is a plausible continuation of the first sentence.

### Why BERT?

BERT represents a major transition from traditional sequential NLP models toward **Transformer-based language models**.

It allows the model to capture contextual relationships between words using **Self-Attention**, rather than relying solely on sequential processing as in traditional RNNs.

---

# 🧠 NLP Tasks Covered

The repository currently covers several important Natural Language Processing tasks.

| Task                         | Status      | Project                  | Main Approach      |
| ---------------------------- | ----------- | ------------------------ | ------------------ |
| Text Classification          | ✅ Completed | SMS Spam Detection       | Neural Network     |
| Spam Detection               | ✅ Completed | SMS Spam Detection       | TensorFlow / Keras |
| Hate Speech Detection        | ✅ Completed | Hate Speech Detection    | Deep Learning      |
| Sentiment Analysis           | ✅ Completed | Sentiment Analysis       | RNN                |
| Sequence Modeling            | ✅ Completed | Sentiment Analysis       | RNN                |
| Sentence Pair Classification | ✅ Completed | Next Sentence Prediction | BERT               |
| Transformer-based NLP        | ✅ Completed | Next Sentence Prediction | BERT               |
| Pretrained Language Models   | ✅ Completed | Next Sentence Prediction | BERT               |

---

# 📂 Repository Structure

```text
Deep-Learning-in-NLP/
│
├── data/
│
├── Hate_Speech_Detection.ipynb
├── Next_Sentence_Prediction.ipynb
├── Sentiment_Analysis.ipynb
├── SMS_Spam_Detection.ipynb
│
├── requirements.txt
├── README.md
└── myvenv/
```

---

# 📊 Datasets

Different datasets are used depending on the NLP task.

## SMS Spam Dataset

Used for the SMS Spam Detection project.

The dataset contains text messages labeled according to whether they are **spam or legitimate messages**.

### Hate Speech Dataset

Used for the Hate Speech Detection project.

The dataset contains textual samples labeled according to their corresponding category.

### Sentiment Dataset

Used for the Sentiment Analysis project.

The dataset contains text samples with sentiment labels that allow the model to learn how linguistic patterns correspond to different sentiment categories.

### Sentence Pair Dataset

Used for the Next Sentence Prediction project.

The dataset contains pairs of sentences that can be used to train or evaluate whether the second sentence follows the first sentence.

---

# 🛠️ Technologies

The projects are primarily implemented using Python and popular Machine Learning and NLP libraries.

## Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras
* PyTorch

### NLP / Transformers

* Hugging Face Transformers
* BERT
* Tokenizers

### Machine Learning

* Scikit-learn

### Data Processing

* NumPy
* Pandas

### Visualization

* Matplotlib

### Development Environment

* Jupyter Notebook

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/Deep-Learning-in-NLP.git
cd Deep-Learning-in-NLP
```

Create a virtual environment:

```bash
python3 -m venv myvenv
```

Activate the environment.

## Linux / macOS

```bash
source myvenv/bin/activate
```

### Windows

```bash
myvenv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

---

# ▶️ Usage

Each project is implemented as an individual Jupyter Notebook.

For example:

```text
SMS_Spam_Detection.ipynb
```

can be opened and executed step-by-step.

The general NLP workflow used throughout the repository is:

```text
Raw Text
   ↓
Text Cleaning
   ↓
Tokenization
   ↓
Vocabulary / Token IDs
   ↓
Text Vectorization
   ↓
Embedding / Contextual Representation
   ↓
Deep Learning Model
   ↓
Training
   ↓
Evaluation
   ↓
Prediction
```

Depending on the project, some stages may be different.

For example, Transformer-based models such as BERT use specialized tokenization and contextual representations instead of the traditional Embedding → RNN pipeline.

---

# 🗺️ Learning Path

The projects are organized to provide a gradual progression from fundamental NLP concepts toward modern Transformer-based approaches.

```text
Text Data
    ↓
Text Preprocessing
    ↓
Tokenization
    ↓
Text Vectorization
    ↓
Word Embeddings
    ↓
Text Classification
    ↓
Sequence Modeling
    ↓
RNNs
    ↓
Sentiment Analysis
    ↓
Attention Mechanism
    ↓
Transformers
    ↓
BERT
    ↓
Modern NLP
```

## Recommended Order

### Step 1 — Text Classification Fundamentals

Start with:

```text
SMS_Spam_Detection.ipynb
```

This project introduces the basic workflow of converting raw text into numerical representations and training a Deep Learning model.

---

#### Step 2 — More Complex Text Classification

Continue with:

```text
Hate_Speech_Detection.ipynb
```

This project applies similar NLP techniques to a more challenging text classification problem.

---

#### Step 3 — Sequential NLP

Then study:

```text
Sentiment_Analysis.ipynb
```

This introduces **Recurrent Neural Networks** and sequence modeling for natural language.

---

#### Step 4 — Transformers and BERT

Finally explore:

```text
Next_Sentence_Prediction.ipynb
```

This introduces the **Transformer architecture, Self-Attention, BERT, and pretrained language models**.

---

# 🎯 Learning Objectives

The main goal of this repository is to develop practical knowledge of **Deep Learning for Natural Language Processing**.

By working through these projects, you can learn how to:

* Process and clean textual data.
* Perform text preprocessing.
* Tokenize natural language.
* Build vocabularies.
* Convert text into numerical representations.
* Use text vectorization.
* Understand word embeddings.
* Build Deep Learning models for text classification.
* Perform binary and multi-class text classification.
* Detect spam messages.
* Detect hate speech.
* Perform sentiment analysis.
* Understand sequential data.
* Build and train RNN-based models.
* Understand the limitations of traditional sequential models.
* Understand Attention Mechanisms.
* Understand Transformer architectures.
* Work with BERT.
* Use pretrained language models.
* Perform sentence pair classification.
* Build end-to-end NLP pipelines.

---

# 🔬 Future Improvements

The repository can be extended with more advanced NLP architectures and applications.

Potential future projects include:

* [ ] LSTM-based Text Classification
* [ ] GRU-based Text Classification
* [ ] Bidirectional LSTM
* [ ] Attention Mechanism
* [ ] Encoder-Decoder Architecture
* [ ] Seq2Seq Models
* [ ] Machine Translation
* [ ] Named Entity Recognition (NER)
* [ ] Question Answering
* [ ] Text Summarization
* [ ] Text Generation
* [ ] Transformer from Scratch
* [ ] BERT Fine-Tuning
* [ ] GPT-based Text Generation
* [ ] Retrieval-Augmented Generation (RAG)
* [ ] Large Language Model Applications
* [ ] NLP Model Deployment

---

# 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

If you would like to contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Commit your changes.
5. Open a Pull Request.

---

# ⭐ Support

If you find this repository useful for learning **Deep Learning and Natural Language Processing**, consider giving it a ⭐ on GitHub.

---

# 👨‍💻 Author

**Soheil Hooshmand**

Computer Engineering — Amirkabir University of Technology

This repository is part of my ongoing learning and development in:

**Machine Learning → Deep Learning → Natural Language Processing → Artificial Intelligence**

---

## 📌 Summary

This repository follows a practical progression from fundamental NLP tasks to modern Transformer-based language models:

```text
Text Classification
        ↓
Text Preprocessing
        ↓
Tokenization & Vectorization
        ↓
Word Embeddings
        ↓
Deep Learning
        ↓
RNNs
        ↓
Sentiment Analysis
        ↓
Attention
        ↓
Transformers
        ↓
BERT
        ↓
Modern NLP
```

The ultimate goal is to build a strong practical foundation for developing **real-world Deep Learning and NLP systems**.
