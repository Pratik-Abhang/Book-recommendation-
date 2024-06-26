import pickle
import streamlit as st 
import numpy as np 
import pandas as pd 

st.header("Book Recommendation App")
model = pickle.load(open(r'C:\Users\PRATIK\Book_Recommender/model.pkl', 'rb'))
book_name = pickle.load(open(r'C:\Users\PRATIK\Book_Recommender/book_name.pkl', 'rb'))
final_rating = pickle.load(open(r'C:\Users\PRATIK\Book_Recommender/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open(r'C:\Users\PRATIK\Book_Recommender/book_pivot.pkl', 'rb'))

selected_books = st.selectbox("Type or Select a Book ", book_name )
