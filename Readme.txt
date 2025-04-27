
So long, and thanks for all the phish!

A machine learning project to detect phishing emails — combined with an interactive Streamlit web app.

# Features

- Intelligent detection of phishing emails based on text content and URL patterns
- Interactive and user-friendly interface powered by Streamlit
- Visual highlights of suspicious keywords
- Model evaluation with classification report, confusion matrix, and ROC curve

# Built With

- Python 3.10+
- Streamlit - for the interactive web app
- Scikit-Learn -for machine learning
- Joblib - for model serialization
- Pandas - for data handling
- tqdm - for progress tracking

# Getting Started

Clone the repository and install the dependencies:

git clone https://github.com/your-username/phishing-detector.git
cd phishing-detector
pip install -r requirements.txt
streamlit run app.py


# Project Structure

P/
├── models/
│   ├── phishing_model.joblib
│   └── tfidf_vectorizer.joblib
├── assets/
│   └── logo.png
├── data/
│   └── (Training CSV files)
├── app.py
├── phishing_detector.py
├── requirements.txt
└── README.md

# How It Works

The system uses a TF-IDF vectorizer to process email text and a custom URL-count feature to detect suspicious patterns.
A Naive Bayes classifier is trained on the processed features to distinguish between ham and phishing emails.

Training and evaluation are performed in phishing_detector.py, including model visualization.

# To Do

- Add features like attachment size analysis or phishing-specific keyword detection
- Deploy the app on a cloud platform (e.g., Streamlit Cloud, AWS, Heroku)

# Disclaimer

This project is intended for demonstration and educational purposes only.
No guarantees are made regarding the detection of real-world phishing attempts.

# About

Created with curiosity and passion by Simon.
Follow me on GitHub for more projects and ideas!

