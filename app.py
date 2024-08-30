import pandas as pd
import neattext.functions as nfx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the cleaned data
df = pd.read_csv("udemy_courses_clean.csv")

# Initialize vectorizer and create matrix
count_vect = CountVectorizer()
cv_mat = count_vect.fit_transform(df['clean_course_title'])

# Compute cosine similarity matrix
cosine_sim_mat = cosine_similarity(cv_mat)

# Create course indices mapping
course_indices = pd.Series(df.index, index=df['course_title']).drop_duplicates()

# Define recommendation function
def recommend_course(title, num_of_rec=10):
    if title not in course_indices:
        return pd.DataFrame({'course_title': [], 'similarity_scores': []})
    
    # Get the index for the given course title
    idx = course_indices[title]
    
    # Get similarity scores for the course at index `idx`
    scores = list(enumerate(cosine_sim_mat[idx]))
    
    # Sort the scores in descending order
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    # Select course indices and similarity scores
    selected_course_indices = [i[0] for i in sorted_scores[1:num_of_rec+1]]
    selected_course_scores = [i[1] for i in sorted_scores[1:num_of_rec+1]]
    
    # Create a DataFrame with the recommended course titles and similarity scores
    result = df['course_title'].iloc[selected_course_indices]
    rec_df = pd.DataFrame({
        'course_title': result,
        'similarity_scores': selected_course_scores
    })
    
    return rec_df

# Streamlit app
st.title("Course Recommendation System")

# Input for course title
title = st.text_input("Enter course title", "")

# Number of recommendations
num_of_rec = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)

if title:
    recommendations = recommend_course(title, num_of_rec)
    
    if not recommendations.empty:
        st.write(f"Recommendations for '{title}':")
        st.dataframe(recommendations)
    else:
        st.write("Course title not found in the dataset.")
