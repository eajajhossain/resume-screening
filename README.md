# AI Resume Screening and Candidate Ranking System

## Overview

Hiring teams often receive hundreds of resumes for a single job role. Manually reviewing each resume is time-consuming, inconsistent, and inefficient.

This project builds a Machine Learning–based resume screening system that automatically analyzes resumes, compares them with a job description, and ranks candidates based on their relevance to the role.

The system extracts skills from resumes and job descriptions using Natural Language Processing (NLP), calculates similarity scores, and highlights missing skills required for the job.

This project demonstrates how machine learning can assist recruiters by reducing manual workload and improving candidate shortlisting.

---

## Key Features

* Resume text preprocessing using NLP
* Skill extraction from resumes and job descriptions
* TF-IDF vectorization for textual similarity
* Cosine similarity for resume-job matching
* Skill vector comparison
* Candidate ranking based on overall similarity
* Identification of missing skills
* Visualizations for candidate comparison
* Simple UI for uploading resumes and job descriptions

---

## Project Workflow

The system follows the following pipeline:

1. Load resume and job description datasets
2. Preprocess text
3. Extract relevant skills
4. Convert text into TF-IDF vectors
5. Convert skills into binary skill vectors
6. Compute similarity scores
7. Rank resumes based on similarity
8. Identify missing skills for each candidate
9. Display results and visualizations

---

## Technologies Used

### Programming Language

Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* NLTK
* Matplotlib
* Seaborn
* Streamlit

### Machine Learning Techniques

* TF-IDF Vectorization
* Cosine Similarity
* NLP Text Processing

---

## Dataset

The project uses publicly available datasets:

Resume Dataset
https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

Job Description Dataset
https://www.kaggle.com/datasets/PromptCloudHQ/us-jobs-on-monstercom

These datasets provide real-world resume text and job descriptions used to simulate a hiring scenario.

---

## Example Output

The system ranks candidates based on their relevance to a job description.

### Candidate Ranking Example

| Resume ID | Similarity Score |
| --------- | ---------------- |
| 102       | 0.83             |
| 56        | 0.79             |
| 210       | 0.76             |

### Skill Gap Example

Job Skills
Python, Machine Learning, SQL, AWS

Resume Skills
Python, Machine Learning

Missing Skills
SQL, AWS

---

## How to Run the Project

Clone the repository

git clone https://github.com/yourusername/resume-screening-system.git

Navigate to the project directory

cd resume-screening-system

Install required libraries

pip install -r requirements.txt

Run the notebook

jupyter notebook

Run the UI

streamlit run app.py

---

## Project Structure

resume-screening-system

data
├── Resume.csv
└── monster_com-job_sample.csv

notebooks
└── resume_screening_pipeline.ipynb

app.py
requirements.txt
README.md

---

## Future Improvements

* BERT embeddings for semantic similarity
* Named Entity Recognition for better skill extraction
* Support for PDF resume uploads
* Advanced ranking models
* Full recruitment dashboard

---


