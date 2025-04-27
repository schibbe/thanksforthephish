import streamlit as st
import joblib
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

st.set_page_config(page_title="So long, and thanks for all the phish!", page_icon=":guardsman:", layout="wide")

with st.spinner("Catching phish..."):
    model = joblib.load('models/phishing_model.joblib')
    vectorizer = joblib.load('models/tfidf_vectorizer.joblib')

suspicious_words = ["urgent", "limited time", "winner", "download", "prize", "click", "sensitive", "account", "login", "verify", "confirm"]

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  
    text = text.lower()  
    return text

def count_urls(text):
    if isinstance(text, str):
        return len(re.findall(r'http[s]?://\S+', text))
    else:
        return 0

def predict_phishing(text):
    text = preprocess_text(text)
    
    vectorized_input = vectorizer.transform([text])
    
    url_count = count_urls(text)
    
    combined_input = hstack([vectorized_input, [[url_count]]])
    
    prediction = model.predict(combined_input)
  
    phishing_probability = model.predict_proba(combined_input)[0][1]
    
    return prediction[0], phishing_probability

def highlight_suspicious_words(text):
    global suspicious_words  
    for word in suspicious_words:
        text = re.sub(rf'({re.escape(word)})', r'<span style="color:red">\1</span>', text, flags=re.IGNORECASE)
    
    new_words = set(re.findall(r'\b\w+\b', text))  
    suspicious_words.extend(new_words)  
    
    suspicious_words = list(set(suspicious_words))
    
    return text

def explain_probability(probability):
    if probability > 0.8:
        return "Don't panic!"
    elif probability > 0.5:
        return "Not sure, but there are signs."
    else:
        return "Always carry a towel!"

st.markdown("<h1 style='text-align: center; font-size: 32px;'>So long, and thanks for all the phish!</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 6, 1])  
with col2:
    st.image('assets/logo.png', width=60)  

with col2:
    email_content = st.text_area("Let Deep Thought check your E-Mail:", height=150, max_chars=2000, key="email_input")  


with col2:
    if st.button("Propose"):
        if email_content:
            highlighted_text = highlight_suspicious_words(email_content)
                   
            st.markdown(f"**Your checked Mail:**")
            st.markdown(f"<div style='text-align:center'>{highlighted_text}</div>", unsafe_allow_html=True)
                
            result, phishing_probability = predict_phishing(email_content)
            
            if result == 1:
                st.error(f"This E-Mail ist Phishy! Probability: {phishing_probability*100:.2f}%")
            else:
                st.success(f"Law & Order. Probability: {phishing_probability*100:.2f}%")
            
            st.info(explain_probability(phishing_probability))
        else:
            st.warning("Enter some Text...")


st.markdown("<h5 style='text-align: center; font-size: 14px;'>Made with curiosity by someone</h5>", unsafe_allow_html=True)
