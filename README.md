# Customer Segmentation Analysis for Strategic Marketing

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange?style=for-the-badge&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0.3-150458?style=for-the-badge&logo=pandas)
![Project Status](https://img.shields.io/badge/Status-Complete-green.svg?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

An end-to-end data science project focused on classifying customer segments using supervised learning and identifying hidden customer groups through unsupervised clustering. **This project serves as the Capstone for the "Machine Learning with Python Scikit-Learn" course by Bisa AI.**

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

This project simulates a real-world business challenge where an automobile company aims to penetrate new markets. Based on their success with customer segmentation in existing markets, this analysis leverages machine learning to automate the segmentation process for new customers. The goal is to ensure marketing efforts are data-driven, efficient, and highly effective.

The project explores two primary machine learning paradigms:
*   **Supervised Learning:** To predict the pre-defined customer segments (A, B, C, D) using various classification algorithms.
*   **Unsupervised Learning:** To discover natural groupings within the customer base without relying on pre-existing labels, potentially revealing new and untapped customer personas.

---

## ❓ Problem Statement

The core task is to develop a predictive model that accurately assigns new customers to one of the four existing segments (A, B, C, or D). The project aims to answer the following key questions:
1.  Which customer attributes (e.g., age, profession, spending score) are most influential in determining customer segmentation?
2.  Which machine learning model (among SVM, Decision Tree, and Gaussian Naive Bayes) provides the best performance for this classification task?
3.  Can unsupervised clustering techniques (K-Means, DBSCAN) reveal meaningful customer personas that align with or differ from the existing segments, providing deeper marketing insights?

---

## 💾 Dataset

The dataset used in this project is the "Customer Segmentation" dataset from the Analytics Vidhya Janatahack contest, publicly available on Kaggle.

*   **Source:** [Kagle Customer Segmentation Dataset](https://www.kaggle.com/datasets/vetrirah/customer)
*   **Size:** 8,068 rows × 11 columns
*   **Key Attributes:** `Gender`, `Age`, `Profession`, `Work_Experience`, `Spending_Score`, `Family_Size`
*   **Target Variable:** `Segmentation`

---

## 🛠️ Methodology

The project follows a structured data science workflow:

1.  **Data Cleaning & Preprocessing:**
    *   Handled outliers in numerical features using the IQR capping method.
    *   Imputed missing values using statistical measures (mode for categorical, median for numerical).
    *   Engineered a new feature, `Age_Married_Category`, by combining `Age` and `Ever_Married` status to create richer personas.
    *   Encoded categorical variables using Ordinal Encoding for features with inherent order (`Spending_Score`) and One-Hot Encoding for nominal features.
    *   Applied feature scaling (`StandardScaler`) to normalize numerical data for distance-based algorithms like SVM and K-Means.

2.  **Exploratory Data Analysis (EDA):**
    *   Analyzed the distribution of the target variable and all predictive features.
    *   Visualized relationships between different customer attributes (univariate, bivariate, and multivariate analysis) to uncover initial patterns.

3.  **Supervised Modeling (Classification):**
    *   Implemented and trained three classification models: **Support Vector Machine (SVM)**, **Decision Tree**, and **Gaussian Naive Bayes**.
    *   Evaluated models based on **Accuracy, Precision, Recall, and F1-Score**. Analyzed performance using a Confusion Matrix for each model.

4.  **Unsupervised Modeling (Clustering):**
    *   Utilized the **Elbow Method** to determine the optimal number of clusters for K-Means.
    *   Applied **K-Means** and **DBSCAN** algorithms to group customers based on their intrinsic characteristics.
    *   Analyzed and interpreted the resulting clusters to define distinct, data-driven customer personas.

---

## ✨ Key Findings & Results

### Classification: SVC Leads with Moderate Performance

*   The **Support Vector Machine (SVC)** model achieved the highest accuracy of **52.9%** on the test set, outperforming the Decision Tree (42.6%) and Gaussian Naive Bayes (48.7%) models.
*   While a formal feature importance analysis was not performed, the Exploratory Data Analysis suggested that **`Age`**, **`Spending_Score`**, **`Ever_Married`**, and **`Profession`** were significant drivers in distinguishing between segments.
*   The moderate accuracy across all models indicates that the pre-defined segments (A, B, C, D) have complex, overlapping boundaries that are challenging to separate with the given features alone.

![Model Comparison (Supervised)]([reports/figures/model_accuracy_comparison.png](https://github.com/LatiefDataVisionary/customer-segmentation-strategy/blob/main/reports/figures/Comparison_of_Classification_Model_Accuracy_and_F1-Score_(Macro_Avg).png))

### Clustering: Five Distinct Data-Driven Personas Identified

*   The Elbow Method suggested an optimal number of **5 clusters** for the K-Means algorithm.
*   The following data-driven customer personas were defined based on an analysis of each cluster's average characteristics:
    *   **Cluster 0: Middle-aged, Married Male Artists (Average Spenders)**
    *   **Cluster 1: Middle-aged, Single Female Artists (Low Spenders)**
    *   **Cluster 2: Young, Single Males with Large Families (Low Spenders, often in Healthcare)**
    *   **Cluster 3: Middle-aged, Married Female Artists (High Work Experience, Low Spenders)**
    *   **Cluster 4: Senior, Married Male Artists (Low Work Experience, Low Spenders)**
*   In contrast, the **DBSCAN** algorithm classified a large portion of the data (~63% of the training set) as noise. This suggests that the customer data does not form distinct high-density regions, making partition-based clustering like K-Means more suitable for this dataset.

---

## 📁 Repository Structure

This project is organized with a clean, scalable, and best-practice directory structure to ensure maintainability and reproducibility.

*   **customer-segmentation-strategy** (Root Directory)
    *   📂 **data**
        *   📂 **processed**
        *   📂 **unprocessed**
    *   📂 **models**
    *   📂 **notebooks**
    *   📂 **reports**
        *   📂 **figures**
    *   📂 **src**
        *   📂 **utils**
    *   📄 **.gitignore**
    *   📄 **LICENSE**
    *   📄 **README.md**
    *   📄 **requirements.txt**

| Directory / File      | Description                                                                                              |
|-----------------------|----------------------------------------------------------------------------------------------------------|
| **`data`**            | Contains all datasets. Separated into **`unprocessed`** (raw data) and **`processed`** (cleaned data).      |
| **`models`**          | Stores serialized, trained machine learning models (e.g., `.joblib` or `.pkl` files).                       |
| **`notebooks`**       | Houses the primary Jupyter Notebook containing the full workflow from EDA to modeling.                  |
| **`reports/figures`** | Stores all generated visualizations, plots, and charts for easy access and reporting.                 |
| **`src/utils`**       | Contains reusable Python utility scripts (.py) for functions like data preprocessing or custom plots.   |
| **`requirements.txt`**| A list of all necessary Python libraries and their versions to ensure a reproducible environment.    |
| **`README.md`**       | This file. The front page providing a comprehensive overview of the project.                                |
| **`LICENSE`**         | The project's license file (e.g., MIT License) indicating how others can use this code.               |
| **`.gitignore`**      | Specifies which files or folders to ignore in version control (e.g., cache files, virtual environments). |

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
*   **Course Inspiration:** Final Project for the Machine Learning with Scikit-Learn class by [Bisa AI](https://bisa.ai/my_course/detail/1/145263#my-desk)).

