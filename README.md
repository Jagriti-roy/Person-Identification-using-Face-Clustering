#  Person Identification using Face Clustering

An unsupervised face recognition system that automatically groups images of the same individual by extracting facial embeddings with **InsightFace** and clustering them using the **DBSCAN** algorithm.



---

##  Directory Layout

```text
person-identification/
│
├── core/
├── data/
├── output/
├── reports/
├── confidence_scores.csv
├── embeddings.npy
├── filenames.npy
├── requirements.txt
└── README.md
```

---

##  Setup Instructions

Clone the repository:

```bash
git clone <repository-url>
```

Open the project folder:

```bash
cd person-identification
```

Install all required dependencies:

```bash
pip install -r requirements.txt
```

---

##  Tech Stack

- Python
- InsightFace
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib


---

##  Running the Application

Move to the source folder:

```bash
cd core
```

Execute the pipeline:

```bash
python runner.py
```

---

##  Generated Outputs

After execution, the following files and folders are created automatically:

- **Grouped face images** (`output/`)
- **Confidence score report** (`confidence_scores.csv`)
- **Similarity matrix**
- **Performance summary** (`reports/summary.txt`)
- **Evaluation metrics** (`reports/metrics.csv`)
- **Interactive dashboard** (`reports/index.html`)
- **Cluster gallery visualizations** (`reports/cluster_gallery/`)

---


##  Project Workflow

```text
Input Images
      │
      ▼
Face Detection
      │
      ▼
Face Selection
      │
      ▼
Feature Extraction (512-D)
      │
      ▼
Embedding Normalization
      │
      ▼
Cosine Similarity
      │
      ▼
DBSCAN Clustering
      │
      ▼
Confidence Estimation
      │
      ▼
Dashboard & Reports
```

---

##  Overview

This project demonstrates an end-to-end face clustering pipeline capable of automatically organizing images belonging to the same person without requiring predefined identity labels. The generated reports, evaluation metrics, and visualization dashboard provide a complete overview of clustering performance.

---

##  Developed By

**JAGRITI KUMARI**
---
**Btech CSE**
---
**12022002001156**
