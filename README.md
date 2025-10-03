# TESSA: Tweet Emotion & Sentiment Signal Analyzer

**A hands-on learning project to build a deep learning system that discerns positive and negative sentiment from Twitter data.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Status](https://img.shields.io/badge/Status-Learning%20Project-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Project Overview

TESSA is a personal portfolio project created to explore and implement a complete end-to-end Natural Language Processing (NLP) pipeline, from data preprocessing to training and evaluating a sophisticated deep learning model.

The project uses Twitter sentiment analysis as a practical case study to tackle the challenges of working with noisy, unstructured text. The primary goal was to gain hands-on experience with technologies like **Word2Vec** and **Bi-LSTMs** and to build a functional and accurate sentiment classification engine from the ground up. The project is relevant as "public opinion on platforms like Twitter is a goldmine of information for businesses" to make data-driven decisions.

## Key Skills & Features Demonstrated

* **High-Accuracy Sentiment Classification:** Successfully built a model that achieved **91% accuracy** in distinguishing between positive and negative sentiments.
* **Advanced NLP Preprocessing:** Implemented a comprehensive text cleaning pipeline to handle the noisy nature of Twitter data (e.g., URLs, hashtags, mentions, stop words).
* **Semantic Word Embeddings:** Utilized **Word2Vec** to generate vector representations of words, capturing the semantic context and relationships between them to improve model performance.
* **Deep Learning Model:** Built and trained a **Bi-directional Long Short-Term Memory (Bi-LSTM)** network, demonstrating an understanding of recurrent neural networks for sequential data.

## Tech Stack & Libraries

* **Language:** Python
* **Core Data Science Libraries:** Pandas, NumPy, Scikit-learn (for metrics and evaluation)
* **NLP & Deep Learning:** TensorFlow (Keras), NLTK, Gensim (for Word2Vec)
* **Data Visualization:** Matplotlib, Seaborn
* **Development Environment:** Jupyter Notebooks, Git

## Project Workflow

The project follows a systematic machine learning pipeline:

1.  **Data Ingestion:** Loaded a dataset of 10,000+ tweets with pre-labeled sentiments.
2.  **Text Preprocessing & Cleaning:** Applied a series of NLP techniques to clean and normalize the text data.
3.  **Feature Extraction (Word Embeddings):** Trained a Word2Vec model on the cleaned corpus to generate dense vector embeddings for each word.
4.  **Model Building:** Constructed a Bi-LSTM model using Keras, designed to process sequential data effectively.
5.  **Model Training & Evaluation:** Trained the model on the processed data and evaluated its performance using metrics like accuracy, precision, recall, and F1-score.
6.  **Sentiment Prediction:** The final trained model is capable of predicting the sentiment of new, unseen tweets.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

* Python 3.8+
* pip & venv

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/TESSA.git](https://github.com/your-username/TESSA.git)
    cd TESSA
    ```

2.  **Create and activate a virtual environment:**
    * On macOS/Linux:
        ```sh
        python3 -m venv venv
        source venv/bin/activate
        ```
    * On Windows:
        ```sh
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## 📊 Performance

The Bi-LSTM model achieved a **91% accuracy** on the test set, demonstrating a successful application of deep learning techniques for this NLP task.

* **Accuracy:** 91%
* **Precision:** [Add Your Precision Score]
* **Recall:** [Add Your Recall Score]
* **F1-Score:** [Add Your F1-Score]

*(You can add a confusion matrix image here once you have it)*
`![Confusion Matrix](reports/figures/confusion_matrix.png)`

## 🔮 Future Learning & Improvements

As this project was focused on learning, there are several exciting avenues for future development and further skill-building:

* **Deployment:** Containerize the application with Docker and deploy the model as a REST API using Flask or FastAPI to learn MLOps principles.
* **Experiment with Transformers:** Fine-tune a pre-trained transformer model like BERT or RoBERTa to compare its performance against the Bi-LSTM.
* **Expand to Multi-Class Classification:** Enhance the model to detect more nuanced emotions like joy, anger, sadness, or neutral sentiment.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

_This learning project was created by [Lakshya Kumar]._
