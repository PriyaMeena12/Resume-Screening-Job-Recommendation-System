# Resume-Screening-Job-Recommendation-System


# 🧠 Resume Screening & Job Recommendation System

A smart web application that automatically **categorizes resumes**, provides **job role recommendations**, and **extracts candidate information** such as name, contact, email, skills, and education from uploaded resumes (PDF or TXT). Built using **Python, Flask, Scikit-learn, TF-IDF, and PyPDF2**.

---

## 🔍 Features

- 📂 Upload resume files (`.pdf`, `.txt`)
- 🏷️ Predict **Resume Category** (e.g., HR, Data Science, Web Developer)
- 💼 Recommend suitable **Job Role**
- 📝 Extract:
  - Candidate Name
  - Email & Phone Number
  - Skills
  - Education

---

## 🚀 Tech Stack

| Area                  | Tools Used                                     |
|-----------------------|------------------------------------------------|
| Backend & Server      | Flask (Python)                                 |
| Machine Learning      | Scikit-learn, TF-IDF Vectorizer, Random Forest |
| File Parsing          | PyPDF2, Regex                                  |
| Frontend              | HTML, CSS (Responsive UI)                      |
| Model Persistence     | Pickle                                         |

---

## 📸 Screenshots

![App Screenshot](screenshots/demo.png)

---

## 🧪 How it Works

1. Preprocess and clean resumes using custom `cleanResume()` function.
2. Extract text from uploaded PDF/TXT using `PyPDF2`.
3. Use **TF-IDF Vectorizer** and trained **RandomForestClassifier** to:
   - Predict **Resume Category**
   - Recommend **Job Role**
4. Extract additional details using regular expressions (regex).

---

## 🧠 ML Models

- **Resume Categorization Model**  
  Trained on labeled resume data using TF-IDF + RandomForest.

- **Job Recommendation Model**  
  Separate classifier trained on categorized resume data.


---

## 🛠️ Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/resume-screening-job-recommendation.git
cd resume-screening-job-recommendation


## Install dependencies:


pip install -r requirements.txt

## Run the Flask app:

python app.py
