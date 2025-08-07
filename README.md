# 📊 Comparative Information Retrieval Systems: TF-IDF vs. Word2Vec

This project implements two distinct information retrieval (IR) systems to fetch relevant documents for a given query from the LISA dataset. The primary goal is to build, evaluate, and compare a classical keyword-based approach (TF-IDF) with a modern semantic vector-based approach (Word2Vec).

## 📝 Project Overview

The project is structured into two main models that are built and evaluated against each other:

1.  **Classical IR with TF-IDF**: A system that represents documents as vectors of term weights based on frequency and rarity. It includes dimensionality reduction using Latent Semantic Indexing (LSI) for visualization.
2.  **Vector-Based IR with Word2Vec**: A system that learns semantic embeddings for words using a Skip-Gram model. Documents are represented by averaging the vectors of their words.

---

## ✨ Part 1: TF-IDF & LSI Retrieval System

This model follows a traditional pipeline for information retrieval.

### Methodology
1.  **Preprocessing**: Text from documents and queries was cleaned by lowercasing, removing punctuation and digits, tokenizing, and filtering out common English stopwords using `nltk`.
2.  **Vectorization**: Documents were converted into numerical vectors using the **TF-IDF (Term Frequency-Inverse Document Frequency)** weighting scheme.
3.  **Retrieval**: The **Cosine Similarity** between a query vector and all document vectors was calculated to rank documents. The top-K most similar documents were retrieved.
4.  **Evaluation**: Performance was measured using **Precision, Recall, and F1-Score** against the provided ground truth for K values of 5, 10, 20, and 40.
5.  **Dimensionality Reduction**: **Latent Semantic Indexing (LSI)** was applied to reduce the TF-IDF vectors to 2 dimensions for visualization.

### Key Finding
-   The F1-Score peaked around K=10, after which Precision dropped more significantly than Recall increased. This is a classic trade-off in IR systems.

---

## ✨ Part 2: Word2Vec Retrieval System

This model leverages dense vector embeddings to capture semantic meaning.

### Methodology
1.  **Model Training**: A **Word2Vec (Skip-Gram)** model was trained on the entire preprocessed text corpus to generate 100-dimensional word embeddings using `gensim`.
2.  **Vector Representation**: Each document and query was transformed into a single vector by **averaging the Word2Vec embeddings** of its constituent words.
3.  **Retrieval**: Cosine similarity was used to rank and retrieve the most similar document vectors for each query vector.

---

## 📈 Comparison and Final Results

The two systems were directly compared using standard IR metrics.


### Metric Summary Table

| K | Metric | TF-IDF | Word2Vec (Avg) |
|---|---|---|---|
| 5 | Precision | **0.1143** | 0.0857 |
| 5 | Recall | **0.2286** | 0.1643 |
| 5 | F1-Score | **0.1448** | 0.1080 |
| 10 | Precision | **0.0943** | 0.0743 |
| 10 | Recall | **0.3310** | 0.2500 |
| 10 | F1-Score | **0.1423** | 0.1116 |
| 20 | Precision | 0.0614 | **0.0629** |
| 20 | Recall | **0.4119** | 0.3952 |
| 20 | F1-Score | 0.1051 | **0.1070** |
| 40 | Precision | 0.0393 | **0.0407** |
| 40 | Recall | 0.4976 | **0.4976** |
| 40 | F1-Score | 0.0722 | **0.0748** |

### Conclusion
The **TF-IDF model generally outperformed the Word2Vec averaging method**, especially in terms of Precision and F1-Score at lower K values (retrieving fewer, more relevant documents). The simple averaging of word vectors likely diluted the signal of important keywords, making it less precise. However, Word2Vec's recall eventually caught up, suggesting it captured some semantic relationships that TF-IDF missed. For a precise retrieval task on this dataset, TF-IDF was the more effective implementation.

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

## 🛠️ Technologies Used

-   Python
-   Pandas
-   NLTK (for preprocessing)
-   Scikit-learn (for TF-IDF, LSI, and metrics)
-   Gensim (for Word2Vec)
-   Matplotlib & Seaborn (for plotting)
-   Jupyter Notebook
-   NumPy
