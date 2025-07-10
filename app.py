from flask import Flask, request, render_template
from PyPDF2 import PdfReader
import re
import pickle
import os

app = Flask(__name__)

# Load models with error handling
try:
    rf_classifier_categorization = pickle.load(open('models/rf_classifier_categorization.pkl', 'rb'))
    tfidf_vectorizer_categorization = pickle.load(open('models/tfidf_vectorizer_categorization.pkl', 'rb'))
except FileNotFoundError:
    rf_classifier_categorization = None
    tfidf_vectorizer_categorization = None

try:
    rf_classifier_job_recommendation = pickle.load(open('models/rf_classifier_job_recommendation.pkl', 'rb'))
    tfidf_vectorizer_job_recommendation = pickle.load(open('models/tfidf_vectorizer_job_recommendation.pkl', 'rb'))
except FileNotFoundError:
    rf_classifier_job_recommendation = None
    tfidf_vectorizer_job_recommendation = None

# Resume cleaner
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText

def predict_category(resume_text):
    if rf_classifier_categorization and tfidf_vectorizer_categorization:
        resume_text = cleanResume(resume_text)
        resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
        return rf_classifier_categorization.predict(resume_tfidf)[0]
    return "Model not available"

def job_recommendation(resume_text):
    if rf_classifier_job_recommendation and tfidf_vectorizer_job_recommendation:
        resume_text = cleanResume(resume_text)
        resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
        return rf_classifier_job_recommendation.predict(resume_tfidf)[0]
    return "Model not available"

def pdf_to_text(file):
    reader = PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ''
    return text

def extract_contact_number_from_resume(text):
    match = re.search(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", text)
    return match.group() if match else None

def extract_email_from_resume(text):
    match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
    return match.group() if match else None

def extract_name_from_resume(text):
    match = re.search(r"(\b[A-Z][a-z]+\b)\s(\b[A-Z][a-z]+\b)", text)
    return match.group() if match else None

def extract_skills_from_resume(text):
    skills_list = ['Python', 'Machine Learning', 'Data Analysis', 'SQL', 'Java', 'C++']  # Trimmed for brevity
    found_skills = [skill for skill in skills_list if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE)]
    return found_skills

def extract_education_from_resume(text):
    edu_keywords = ['Computer Science', 'Information Technology', 'Mechanical Engineering']  # Trimmed for brevity
    found_education = [keyword for keyword in edu_keywords if re.search(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE)]
    return found_education

@app.route('/')
def resume():
    return render_template("resume.html")

@app.route('/pred', methods=['POST'])
def pred():
    if 'resume' not in request.files:
        return render_template('resume.html', message="No resume file uploaded.")

    file = request.files['resume']
    if file.filename.endswith('.pdf'):
        text = pdf_to_text(file)
    elif file.filename.endswith('.txt'):
        text = file.read().decode('utf-8')
    else:
        return render_template('resume.html', message="Invalid file format. Upload PDF or TXT.")

    predicted_category = predict_category(text)
    recommended_job = job_recommendation(text)
    phone = extract_contact_number_from_resume(text)
    email = extract_email_from_resume(text)
    name = extract_name_from_resume(text)
    extracted_skills = extract_skills_from_resume(text)
    extracted_education = extract_education_from_resume(text)

    return render_template('resume.html', predicted_category=predicted_category, recommended_job=recommended_job,
                           phone=phone, email=email, name=name, extracted_skills=extracted_skills,
                           extracted_education=extracted_education)

if __name__ == '__main__':
    app.run(debug=True)
