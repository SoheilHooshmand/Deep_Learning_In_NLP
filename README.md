# 📝 Deep Learning in NLP

A hands-on collection of **Deep Learning and Natural Language Processing (NLP) projects** implemented with Python.

This repository explores the application of Deep Learning to **text and natural language data**, starting from fundamental text classification tasks and progressing toward more advanced NLP applications such as **Spam Detection, Hate Speech Detection, Sentiment Analysis, Next Sentence Prediction using BERT, Fine-tuned BERT Sentiment Analysis with Keras Hub, Fine-tuning BERT for Sentiment Analysis with PyTorch, Machine Translation with Transformers, and Simple Chatbot using Vector Embeddings**.

The projects are implemented as Jupyter Notebooks and combine theoretical concepts with practical implementations and experiments using modern Deep Learning frameworks.

---

## 📚 Contents

* [Projects](#-projects)
* [SMS Spam Detection](#1-sms-spam-detection)
* [Hate Speech Detection](#2-hate-speech-detection)
* [Sentiment Analysis with RNN](#3-sentiment-analysis-with-rnn)
* [Next Sentence Prediction using BERT](#4-next-sentence-prediction-using-bert)
* [Sentiment Analysis with BERT (Keras Hub)](#5-sentiment-analysis-with-bert-keras-hub)
* [Fine-tuning BERT for Sentiment Analysis (PyTorch)](#6-fine-tuning-bert-for-sentiment-analysis-pytorch)
* [Machine Translation with Transformer](#7-machine-translation-with-transformer)
* [Simple Chatbot using Vector Embeddings](#8-simple-chatbot-using-vector-embeddings)
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

## 2. Hate Speech Detection

**Notebook:** `Hate_Speech_Detection.ipynb`

A Natural Language Processing project focused on detecting **hate speech in textual data** using Deep Learning.

The model learns linguistic patterns from labeled text and predicts whether a given text belongs to the target hate-speech category.

### Main Concepts

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

## 3. Sentiment Analysis with RNN

**Notebook:** `Sentiment_Analysis.ipynb`

A sentiment analysis project that uses **Recurrent Neural Networks (RNNs)** to classify the sentiment expressed in textual data.

RNN-based architectures are designed to process sequential data and are particularly useful for understanding the order and context of words within a sentence.

### Main Concepts

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

## 4. Next Sentence Prediction using BERT

**Notebook:** `Next_Sentence_Prediction.ipynb`

This project explores **Next Sentence Prediction (NSP)** using **BERT (Bidirectional Encoder Representations from Transformers)**.

Unlike traditional RNN-based approaches, BERT is based on the **Transformer architecture** and uses bidirectional contextual representations to understand relationships between words and sentences.

Next Sentence Prediction is a pretraining task originally introduced with BERT to determine whether one sentence logically follows another.

### Main Concepts

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

## 5. Sentiment Analysis with BERT (Keras Hub)

**Notebook:** `Sentiment_Analysis_BERT_KerasHub.ipynb`

A complete end-to-end **Sentiment Analysis** project using a fine-tuned **BERT** model with **Keras Hub**.

This project works on the classic **IMDB Movie Reviews** dataset and demonstrates how to fine-tune a pretrained BERT model for binary sentiment classification (Positive / Negative).

### Main Concepts

* Transformers
* BERT (Bidirectional Encoder Representations from Transformers)
* Keras Hub
* Fine-tuning pretrained language models
* Text preprocessing & cleaning
* WordCloud visualization
* Transfer Learning
* Binary text classification
* Model saving & loading
* End-to-end NLP pipeline

### General Pipeline

```text
IMDB Movie Reviews
↓
Text Cleaning (HTML removal, special characters, ...)
↓
Exploratory Analysis (WordCloud)
↓
Train / Validation / Test Split
↓
BERT Preprocessor (Keras Hub)
↓
BertTextClassifier (Fine-tuning)
↓
Training (Adam optimizer)
↓
Evaluation + Classification Report
↓
Save / Load Model
↓
Sentiment Prediction Function
```

### Key Features

* Uses `keras_hub.models.BertTextClassifier` with the `bert_base_en_uncased` preset
* Automatic preprocessing with the built-in BERT preprocessor
* Text cleaning with BeautifulSoup + regular expressions
* Positive & Negative WordClouds
* Proper train / validation / test split
* Model saving and reloading
* Ready-to-use `Get_sentiment()` function for new reviews

This project shows a modern and practical way of applying **Transfer Learning** with BERT for real-world sentiment analysis tasks.

---

## 6. Fine-tuning BERT for Sentiment Analysis (PyTorch)

**Notebook:** `fine-tuning_BERT_model_for_Sentiment_Analysis.ipynb`

A complete **Sentiment Analysis** project that fine-tunes a pretrained **BERT** model using **PyTorch** and the Hugging Face Transformers library.

In this project the pretrained `bert-base-uncased` model is loaded, its layers are frozen, and a custom classification head is added on top of the `[CLS]` token representation. The model is then trained for binary sentiment classification.

### Main Concepts

* Transformers
* BERT (Bidirectional Encoder Representations from Transformers)
* Hugging Face Transformers
* PyTorch
* Feature Extraction (Frozen BERT)
* Custom Classification Head
* Tokenization with `BertTokenizerFast`
* Attention Mask
* Transfer Learning
* Binary Sentiment Classification
* Gradient Clipping
* Model Evaluation (Classification Report)

### General Pipeline

```text
Sentiment Dataset (CSV)
↓
Train / Validation / Test Split (stratified)
↓
BERT Tokenizer (max_length = 17)
↓
Convert to PyTorch Tensors (input_ids + attention_mask)
↓
Load bert-base-uncased
↓
Freeze all BERT parameters
↓
Add Custom Head (Linear → ReLU → Dropout → Linear → LogSoftmax)
↓
Training Loop (AdamW + CrossEntropyLoss)
↓
Validation Loop
↓
Test Evaluation + Classification Report
```

### Key Features

* Uses `bert-base-uncased` from Hugging Face
* Freezes the entire BERT backbone (feature extraction approach)
* Custom PyTorch `nn.Module` classification head on top of the `[CLS]` embedding
* Proper stratified train/validation/test split
* Attention masks and padding
* Gradient clipping for stable training
* Full classification report on the test set

This project demonstrates a classic and educational way of performing **Transfer Learning** with BERT using pure PyTorch.

---

## 7. Machine Translation with Transformer

**Notebook:** `Machine_Translation_with_Transformer.ipynb`

A complete **English → Hindi Machine Translation** project based on a pretrained **Seq2Seq Transformer** model (`Helsinki-NLP/opus-mt-en-hi`).

The project demonstrates the full pipeline of loading a parallel corpus, preprocessing, partial fine-tuning (layer freezing), evaluation with **SacreBLEU**, and deploying an interactive translation interface using **Gradio**.

### Main Concepts

* Sequence-to-Sequence (Seq2Seq) Models
* Transformer Architecture (Encoder-Decoder)
* Machine Translation
* Hugging Face Transformers
* MarianMT / OPUS-MT models
* Tokenization for translation
* Fine-tuning pretrained translation models
* Partial Fine-tuning (Layer Freezing)
* Data Collator for Seq2Seq
* SacreBLEU evaluation metric
* Gradient Checkpointing & Mixed Precision (fp16)
* Gradio deployment

### General Pipeline

```text
IITB English-Hindi Dataset
↓
Load Pretrained Model (opus-mt-en-hi)
↓
Tokenization (Source & Target)
↓
Partial Fine-tuning (Freeze lower layers)
↓
Seq2Seq Training (Adafactor + fp16)
↓
Evaluation with SacreBLEU
↓
Interactive Translation Interface (Gradio)
```

### Key Features

* Uses the parallel corpus `cfilt/iitb-english-hindi`
* Loads the strong pretrained model `Helsinki-NLP/opus-mt-en-hi`
* Implements **layer freezing** (keeps only the top layers trainable)
* Uses `DataCollatorForSeq2Seq` for proper padding
* Evaluates translation quality with **SacreBLEU**
* Trains with mixed precision (`fp16`) and gradient checkpointing for efficiency
* Deploys a ready-to-use Gradio web interface for live translation

This project shows a practical and production-oriented approach to fine-tuning modern Transformer-based Machine Translation systems.

---

## 8. Simple Chatbot using Vector Embeddings

**Notebook:** `Simple_Chatbot_using_Vector_Embeddings.ipynb`

A practical project that builds a **conversational chatbot with long-term memory** using vector embeddings and a persistent vector database.

The system stores conversation history as embeddings in **ChromaDB**, retrieves the most relevant past interactions for each new user message via semantic search, and injects them into the prompt of a small language model (`Qwen2.5-0.5B-Instruct`) to generate context-aware responses.

### Main Concepts

* Conversational AI / Chatbot
* Vector Embeddings
* Semantic Search
* Persistent Vector Database (ChromaDB)
* Retrieval-Augmented Generation (simple RAG)
* Sentence Transformers
* Small Language Models
* Prompt Engineering with Retrieved Context
* Multi-turn Dialogue Memory
* Metadata Filtering

### General Pipeline

```text
User Message
↓
Embed Query (SentenceTransformer - all-MiniLM-L6-v2)
↓
Semantic Search in ChromaDB
↓
Retrieve Top-k Relevant Memories
↓
Build Context (Retrieved Memories + Current Question)
↓
Prompt Small LLM (Qwen2.5-0.5B-Instruct)
↓
Generate Response
↓
Store New Interaction (User + Assistant) back into Memory
```

### Key Features

* Uses `all-MiniLM-L6-v2` for efficient sentence embeddings
* Persistent storage with **ChromaDB** (memories survive restarts)
* Metadata support (`user_id`, `conversation_id`, `role`, timestamps)
* Automatic storage of both user messages and assistant replies
* Context construction that includes similarity scores
* Lightweight LLM suitable for educational and low-resource settings
* Interactive chat loop + memory inspection tools

This project serves as an accessible introduction to building memory-augmented chatbots and simple RAG systems.

---

# 🧠 NLP Tasks Covered

The repository currently covers several important Natural Language Processing tasks.

```
| Task                              | Status      | Project                                      | Main Approach                          |
| --------------------------------- | ----------- | -------------------------------------------- | -------------------------------------- |
| Text Classification               | ✅ Completed | SMS Spam Detection                           | Neural Network                         |
| Spam Detection                    | ✅ Completed | SMS Spam Detection                           | TensorFlow / Keras                     |
| Hate Speech Detection             | ✅ Completed | Hate Speech Detection                        | Deep Learning                          |
| Sentiment Analysis                | ✅ Completed | Sentiment Analysis (RNN)                     | RNN                                    |
| Sequence Modeling                 | ✅ Completed | Sentiment Analysis (RNN)                     | RNN                                    |
| Sentence Pair Classification      | ✅ Completed | Next Sentence Prediction                     | BERT                                   |
| Transformer-based NLP             | ✅ Completed | Next Sentence Prediction                     | BERT                                   |
| Pretrained Language Models        | ✅ Completed | Next Sentence Prediction                     | BERT                                   |
| Fine-tuned BERT Sentiment         | ✅ Completed | Sentiment Analysis with BERT (Keras Hub)     | Keras Hub + BERT                       |
| Transfer Learning                 | ✅ Completed | Sentiment Analysis with BERT (Keras Hub)     | Fine-tuning                            |
| Fine-tuning BERT (PyTorch)        | ✅ Completed | Fine-tuning BERT for Sentiment Analysis      | PyTorch + Hugging Face                 |
| Feature Extraction with BERT      | ✅ Completed | Fine-tuning BERT for Sentiment Analysis      | Frozen BERT + Custom Head              |
| Machine Translation               | ✅ Completed | Machine Translation with Transformer         | Seq2Seq Transformer                    |
| Seq2Seq Modeling                  | ✅ Completed | Machine Translation with Transformer         | Encoder-Decoder                        |
| Partial Fine-tuning               | ✅ Completed | Machine Translation with Transformer         | Layer Freezing                         |
| Conversational AI / Chatbot       | ✅ Completed | Simple Chatbot using Vector Embeddings       | Small LLM + Memory                     |
| Retrieval-Augmented Generation    | ✅ Completed | Simple Chatbot using Vector Embeddings       | Embeddings + Vector DB                 |
| Semantic Search                   | ✅ Completed | Simple Chatbot using Vector Embeddings       | Sentence Transformers + ChromaDB       |
| Long-term Dialogue Memory         | ✅ Completed | Simple Chatbot using Vector Embeddings       | Persistent Vector Store                |
```

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
├── Sentiment_Analysis_BERT_KerasHub.ipynb
├── fine-tuning_BERT_model_for_Sentiment_Analysis.ipynb
├── Machine_Translation_with_Transformer.ipynb
├── Simple_Chatbot_using_Vector_Embeddings.ipynb
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
Contains text messages labeled as **spam** or **ham**.

### Hate Speech Dataset
Used for the Hate Speech Detection project.  
Contains textual samples labeled according to their category.

### Sentiment Dataset (RNN)
Used for the RNN-based Sentiment Analysis project.

### Sentence Pair Dataset
Used for the Next Sentence Prediction project.

### IMDB Movie Reviews Dataset
Used for the **Sentiment Analysis with BERT (Keras Hub)** project.  
A large dataset of movie reviews labeled as **positive** or **negative**.

### Sentiment Dataset (PyTorch BERT)
Used for the **Fine-tuning BERT for Sentiment Analysis (PyTorch)** project.  
A labeled sentiment dataset (`sentiment_train.csv`) with sentence and label columns.

### IITB English-Hindi Parallel Corpus
Used for the **Machine Translation with Transformer** project.  
A parallel corpus of English–Hindi sentence pairs (`cfilt/iitb-english-hindi`).

### Conversation Memory (Self-Generated)
Used for the **Simple Chatbot using Vector Embeddings** project.  
The system starts with a few seed memories and continuously stores new user–assistant interactions as vector embeddings in ChromaDB.

---

# 🛠️ Technologies

The projects are primarily implemented using Python and popular Machine Learning and NLP libraries.

## Programming Language
* Python

### Deep Learning
* TensorFlow
* Keras
* Keras Hub
* PyTorch

### NLP / Transformers
* Hugging Face Transformers
* Hugging Face Datasets
* Hugging Face Evaluate (SacreBLEU)
* BERT
* MarianMT / OPUS-MT
* Keras Hub BERT models
* Tokenizers (`BertTokenizerFast`)
* Sentence Transformers

### Vector Databases & Retrieval
* ChromaDB

### Machine Learning
* Scikit-learn

### Data Processing
* NumPy
* Pandas
* BeautifulSoup
* Regular Expressions

### Visualization
* Matplotlib
* Plotly
* WordCloud

### Deployment
* Gradio

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

**Linux / macOS**
```bash
source myvenv/bin/activate
```

**Windows**
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
Simple_Chatbot_using_Vector_Embeddings.ipynb
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
For Transformer-based models such as BERT and Seq2Seq translation models, specialized tokenization and contextual representations are used instead of the traditional Embedding → RNN pipeline.

In the **Simple Chatbot using Vector Embeddings** project, the pipeline becomes:

```text
User Query
↓
Embedding
↓
Vector Retrieval (ChromaDB)
↓
Context Augmentation
↓
LLM Generation
↓
Memory Update
```

---

# 🗺️ Learning Path

The projects are organized to provide a gradual progression from fundamental NLP concepts toward modern Transformer-based and memory-augmented approaches.

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
Fine-tuning BERT (Keras Hub)
↓
Fine-tuning BERT (PyTorch)
↓
Seq2Seq Transformers
↓
Machine Translation
↓
Vector Embeddings + Semantic Search
↓
Retrieval-Augmented Chatbot
↓
Modern NLP Systems
```

## Recommended Order

**Step 1 — Text Classification Fundamentals**  
Start with: `SMS_Spam_Detection.ipynb`

**Step 2 — More Complex Text Classification**  
Continue with: `Hate_Speech_Detection.ipynb`

**Step 3 — Sequential NLP**  
Then study: `Sentiment_Analysis.ipynb`

**Step 4 — Transformers and BERT (NSP)**  
Explore: `Next_Sentence_Prediction.ipynb`

**Step 5 — Fine-tuning BERT for Sentiment Analysis (Keras Hub)**  
Continue with: `Sentiment_Analysis_BERT_KerasHub.ipynb`

**Step 6 — Fine-tuning BERT for Sentiment Analysis (PyTorch)**  
Continue with: `fine-tuning_BERT_model_for_Sentiment_Analysis.ipynb`

**Step 7 — Machine Translation with Seq2Seq Transformers**  
Then: `Machine_Translation_with_Transformer.ipynb`

**Step 8 — Simple Chatbot using Vector Embeddings**  
Finally: `Simple_Chatbot_using_Vector_Embeddings.ipynb`

This final project introduces the practical use of embeddings, vector databases, and building a memory-augmented conversational agent.

---

# 🎯 Learning Objectives

The main goal of this repository is to develop practical knowledge of **Deep Learning for Natural Language Processing**.

By working through these projects, you can learn how to:

* Process and clean textual data
* Perform text preprocessing
* Tokenize natural language
* Build vocabularies
* Convert text into numerical representations
* Use text vectorization
* Understand word embeddings
* Build Deep Learning models for text classification
* Perform binary and multi-class text classification
* Detect spam messages
* Detect hate speech
* Perform sentiment analysis
* Understand sequential data
* Build and train RNN-based models
* Understand the limitations of traditional sequential models
* Understand Attention Mechanisms
* Understand Transformer architectures
* Work with BERT
* Use pretrained language models
* Perform sentence pair classification
* Fine-tune BERT for downstream tasks (both with Keras Hub and pure PyTorch)
* Freeze BERT layers and train a custom classification head
* Use Hugging Face Transformers + PyTorch for BERT fine-tuning
* Use Keras Hub for modern Transformer models
* Build end-to-end NLP pipelines
* Fine-tune Seq2Seq Transformer models for Machine Translation
* Implement partial fine-tuning (layer freezing)
* Evaluate translation quality with SacreBLEU
* Deploy NLP models with Gradio
* Save, load and deploy trained models
* Create high-quality sentence embeddings with Sentence Transformers
* Store and query vectors using ChromaDB
* Implement a simple Retrieval-Augmented Generation (RAG) pipeline
* Build a conversational chatbot with persistent long-term memory
* Manage multi-turn dialogues using metadata filtering

---

# 🔬 Future Improvements

The repository can be extended with more advanced NLP architectures and applications.

Potential future projects include:

* [ ] LSTM-based Text Classification
* [ ] GRU-based Text Classification
* [ ] Bidirectional LSTM
* [ ] Attention Mechanism from Scratch
* [ ] Encoder-Decoder from Scratch
* [ ] Named Entity Recognition (NER)
* [ ] Question Answering
* [ ] Text Summarization
* [ ] Text Generation
* [ ] Transformer from Scratch
* [ ] More BERT Fine-Tuning tasks
* [ ] Full Fine-tuning of BERT (unfreezing all layers)
* [ ] GPT-based Text Generation
* [ ] Advanced Retrieval-Augmented Generation (RAG)
* [ ] Large Language Model Applications
* [ ] Advanced NLP Model Deployment
* [ ] Multi-user Memory Systems
* [ ] Hybrid Search (Keyword + Semantic)

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

This repository follows a practical progression from fundamental NLP tasks to modern Transformer-based and memory-augmented systems:

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
Fine-tuned BERT (Keras Hub)
↓
Fine-tuned BERT (PyTorch)
↓
Seq2Seq Transformers
↓
Machine Translation
↓
Vector Embeddings + Semantic Search
↓
Simple Chatbot with Persistent Memory
↓
Modern NLP
```

The ultimate goal is to build a strong practical foundation for developing **real-world Deep Learning and NLP systems**.
