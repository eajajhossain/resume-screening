import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pdfminer.high_level import extract_text

nltk.download("punkt")
nltk.download("stopwords")


# -----------------------------
# SKILL DATABASE (same as notebook)
# -----------------------------

common_skills = [
 'python', 'java', 'c++', 'javascript', 'html', 'css', 'sql', 'r',
    'data analysis', 'machine learning', 'deep learning', 'artificial intelligence',
    'statistical analysis', 'data mining', 'big data', 'hadoop', 'spark', 'sas',
    'excel', 'tableau', 'power bi', 'aws', 'azure', 'google cloud', 'gcp',
    'docker', 'kubernetes', 'git', 'github', 'jenkins', 'devops',
    'project management', 'agile', 'scrum', 'leadership', 'communication',
    'teamwork', 'problem solving', 'critical thinking', 'creativity',
    'adaptability', 'time management', 'customer service', 'sales',
    'marketing', 'finance', 'accounting', 'human resources', 'hr', 'recruitment',
    'training', 'public speaking', 'negotiation', 'microsoft office', 'word', 'powerpoint',
    'outlook', 'unix', 'linux', 'windows', 'network security', 'cybersecurity',
    'web development', 'mobile development', 'frontend', 'backend', 'full stack',
    'database management', 'cloud computing', 'system administration', 'etl',
    'api development', 'data visualization', 'nlp', 'natural language processing',
    'software development', 'qa testing', 'technical support', 'business analysis',
    'strategic planning', 'risk management', 'financial modeling', 'budgeting'
]


# -----------------------------
# TEXT PREPROCESSING
# -----------------------------

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}', '', text)

    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words("english"))
    tokens = [w for w in tokens if w not in stop_words]

    return " ".join(tokens)


# -----------------------------
# SKILL EXTRACTION
# -----------------------------

def extract_skills(text):

    extracted = []

    for skill in common_skills:
        if skill in text:
            extracted.append(skill)

    return extracted


# -----------------------------
# SKILL VECTOR
# -----------------------------

def get_skill_vector(extracted_skills):

    vector = np.zeros(len(common_skills))

    for i, skill in enumerate(common_skills):
        if skill in extracted_skills:
            vector[i] = 1

    return vector


# -----------------------------
# READ RESUME
# -----------------------------

def read_resume(uploaded_file):

    if uploaded_file.type == "application/pdf":
        text = extract_text(uploaded_file)
    else:
        text = uploaded_file.read().decode()

    return text


# -----------------------------
# STREAMLIT UI
# -----------------------------

st.title("AI Resume Screening System")

st.write("Upload a resume and paste a job description to analyze candidate fit.")

job_description = st.text_area("Paste Job Description")


uploaded_resume = st.file_uploader(
    "Upload Resume",
    type=["pdf", "txt"]
)


if st.button("Analyze Candidate"):

    if uploaded_resume is None or job_description == "":
        st.warning("Please upload resume and paste job description")

    else:

        # read resume
        resume_text = read_resume(uploaded_resume)

        # preprocess
        clean_resume = preprocess_text(resume_text)
        clean_job = preprocess_text(job_description)

        # skill extraction
        resume_skills = extract_skills(clean_resume)
        job_skills = extract_skills(clean_job)

        # skill vectors
        resume_vector = get_skill_vector(resume_skills)
        job_vector = get_skill_vector(job_skills)

        # TFIDF similarity
        tfidf = TfidfVectorizer(max_features=5000)

        tfidf_matrix = tfidf.fit_transform([clean_resume, clean_job])

        text_similarity = cosine_similarity(
            tfidf_matrix[0],
            tfidf_matrix[1]
        )[0][0]

        # skill similarity
        skill_similarity = cosine_similarity(
            [resume_vector],
            [job_vector]
        )[0][0]

        # final score
        final_score = (text_similarity + skill_similarity) / 2

        # missing skills
        missing_skills = [skill for skill in job_skills if skill not in resume_skills]


        st.subheader("Analysis Result")

        st.write("Similarity Score:", round(final_score,3))

        st.write("Job Skills:", job_skills)

        st.write("Candidate Skills:", resume_skills)

        st.write("Missing Skills:", missing_skills)


        # simple bar chart

        chart_df = pd.DataFrame({
            "Metric":["Text Similarity","Skill Similarity"],
            "Score":[text_similarity,skill_similarity]
        })

        st.bar_chart(chart_df.set_index("Metric"))