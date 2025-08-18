# Customer Segmentation Analysis for Strategic Marketing

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange?style=for-the-badge&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0.3-150458?style=for-the-badge&logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

An end-to-end data science project focused on classifying customer segments using supervised learning and identifying hidden customer groups through unsupervised clustering. This analysis helps an automobile company devise targeted marketing strategies for new markets.

---

## 📋 Table of Contents
1.  [**Project Overview**](#-project-overview)
2.  [**Problem Statement**](#-problem-statement)
3.  [**Dataset**](#-dataset)
4.  [**Methodology**](#-methodology)
5.  [**Key Findings & Results**](#-key-findings--results)
6.  [**Repository Structure**](#-repository-structure)
7.  [**Getting Started**](#-getting-started)
8.  [**Tools & Technologies**](#-tools--technologies)
9.  [**Acknowledgements**](#-acknowledgements)

---

## 🎯 Project Overview

This project simulates a real-world business challenge where an automobile company aims to penetrate new markets. Based on their success in existing markets, they have identified the power of customer segmentation. This analysis leverages machine learning to automate the segmentation process for new customers, ensuring that marketing efforts are both efficient and effective.

The project explores two primary machine learning paradigms:
*   **Supervised Learning:** To predict the pre-defined customer segments (A, B, C, D).
*   **Unsupervised Learning:** To discover natural groupings within the customer base, potentially revealing new, untapped segments.

---

## ❓ Problem Statement

The core task is to develop a predictive model that accurately assigns new customers to one of the four existing segments (A, B, C, or D). The project aims to answer the following key questions:
1.  Which customer attributes (e.g., age, profession, family size) are the most influential in determining customer segmentation?
2.  Which machine learning model (among SVM, Decision Tree, and Naive Bayes) provides the best performance for this classification task?
3.  Can unsupervised clustering techniques (K-Means, DBSCAN) reveal meaningful customer personas that align with or differ from the existing segments?

---

## 💾 Dataset

The dataset used in this project is the "Customer Segmentation" dataset from the Analytics Vidhya Janatahack contest, publicly available on Kaggle.

*   **Source:** [Kaggle Customer Segmentation Dataset](https://www.kaggle.com/datasets/vetrirah/customer)
*   **Size:** 8,068 rows × 11 columns
*   **Key Attributes:** `Gender`, `Age`, `Profession`, `Work_Experience`, `Spending_Score`, `Family_Size`
*   **Target Variable:** `Segmentation`

---

## 🛠️ Methodology

The project follows a structured data science workflow:

1.  **Data Cleaning & Preprocessing:**
    *   Handled missing values using statistical imputation (mean, median, mode).
    *   Encoded categorical variables using Label Encoding for ordinal features and One-Hot Encoding for nominal features.
    *   Applied feature scaling (StandardScaler) to normalize the data for distance-based algorithms like SVM and K-Means.

2.  **Exploratory Data Analysis (EDA):**
    *   Analyzed the distribution of customer segments.
    *   Visualized relationships between different customer attributes and the final segment.

3.  **Supervised Modeling (Classification):**
    *   Implemented and trained three classification models:
        *   Support Vector Machine (SVM)
        *   Decision Tree Classifier
        *   Gaussian Naive Bayes
    *   Evaluated models based on Accuracy, Precision, Recall, and F1-Score, and analyzed their performance using a Confusion Matrix.

4.  **Unsupervised Modeling (Clustering):**
    *   Utilized the Elbow Method to determine the optimal number of clusters for K-Means.
    *   Applied K-Means and DBSCAN algorithms to group customers based on their characteristics.
    *   Analyzed and interpreted the resulting clusters to define customer personas.

---

## ✨ Key Findings & Results

This is where you'll summarize your most impactful results.

**Classification:**
*   The **[Your Best Model, e.g., Support Vector Machine]** model achieved the highest accuracy of **[XX.X%]** on the test set.
*   Key features like `[Feature 1]`, `[Feature 2]`, and `[Feature 3]` were identified as the most significant predictors.

*(Optional: Insert a key chart, like your model comparison bar chart)*
![Model Comparison](reports/figures/model_accuracy_comparison.png)

**Clustering:**
*   K-Means clustering identified **[e.g., 4]** distinct customer groups.
*   **Cluster 0 (The Young Professionals):** Characterized by [e.g., younger age, high spending score, and smaller family size]. This group strongly correlates with Segment A.
*   **Cluster 1 (The Established Families):** ...

## 📁 Repository Structure

/
├── data/
│   ├── processed/
│   └── unprocessed/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   └── utils/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

/data/: Contains all datasets, separated into unprocessed (raw data) and processed (cleaned data).
/models/: Stores serialized, trained machine learning models (e.g., .joblib files).
/notebooks/: Houses Jupyter Notebooks for exploration, analysis, and modeling.
/reports/figures/: Stores all generated plots, charts, and figures.
/src/utils/: Contains reusable Python utility scripts for tasks like data preprocessing and visualization.
requirements.txt: A list of all necessary Python packages for easy environment replication.

---

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

*   Python 3.9+
*   pip & virtualenv

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/[YourUsername]/[Your-Repo-Name].git
    cd [Your-Repo-Name]
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install required packages:**
    ```sh
    pip install -r requirements.txt
    ```

### Usage
All the analysis is available in the Jupyter Notebook located at `notebooks/01_customer_segmentation_analysis.ipynb`. Launch Jupyter and open the notebook to view the entire workflow.

```sh
jupyter notebook
```

## 💻 Tools & Technologies

| Tool | Description |
|---|---|
| **Python** | Core programming language for data analysis and modeling. |
| **Pandas & NumPy** | Used for data manipulation, cleaning, and numerical operations. |
| **Matplotlib & Seaborn** | Used for data visualization and exploratory data analysis. |
| **Scikit-learn** | The primary library for implementing machine learning models. |
| **Jupyter Notebook** | Interactive environment for developing and presenting the analysis. |
| **Git & GitHub** | Used for version control and project hosting. |

---

## 🙏 Acknowledgements

*   **Dataset Provider:** [Analytics Vidhya](https://www.analyticsvidhya.com/)
*   **Course Inspiration:** Final Project for the Machine Learning with Scikit-Learn class by [Bisa AI]([https://bisa.ai/](https://bisa.ai/my_course/detail/1/145263#my-desk)).

