import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize  

bengali_punctuation = "[{(%।॥,@&*?!-…‘’“”<#>:;)}]\n"
def bengali_stem(word):
    suffixes = ["গুলো", "গুলি", "দের", "তে", "কে", "রা", "টি", "ে"]
    for suf in suffixes:
        if word.endswith(suf):
           return word[:-len(suf)]
    return word

def processed_text(text):
    tokens = list(indic_tokenize.trivial_tokenize(text, 'bn'))

    cleaned_tokens = [
        bengali_stem(token) for token in tokens
        if token not in bengali_punctuation and token not in set(stopwords.words('bengali'))
    ]
    return ' '.join(cleaned_tokens) if cleaned_tokens else text

def processed_sentence(model, tfidf,input_sms):
    proc_sent = processed_text(input_sms)
    if proc_sent.strip():
        vector_input = tfidf.transform([proc_sent])
        prob = model.predict_proba(vector_input)[0]
        return prob.argmax()
        
    return 0 

tfidf = pickle.load(open('vectorizer1.pkl', 'rb'))
model = pickle.load(open('model1.pkl', 'rb'))


st.title('বার্তাবন্ধু')
input_sms = st.text_area('আপনার বার্তা এখানে লিখুন')
if st.button('এসএমএস শনাক্তকরণ'):
    #1 
    result = processed_sentence(model, tfidf, input_sms)
    
    #4 
    if result == 2:
        st.header('স্প্যাম মেসেজ')
    elif result == 1:
        st.header('বিজ্ঞাপনমূলক মেসেজ')
    else:
        st.header('নরমাল মেসেজ')