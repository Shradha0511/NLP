# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math

st.title("Lab 5: Sparse Vector (Embedding)")

# Sidebar for question selection
option = st.sidebar.selectbox(
    "Select Question",
    ("Question 1: TF-IDF", "Question 2: Cosine Similarity", "Question 3: PMI")
)

# ------------------------
# Question 1: TF-IDF (Pre-filled from lab sheet)
# ------------------------
if option.startswith("Question 1"):
    st.header("TF-IDF Calculation (Using Given Table)")
    
    tf_df = pd.DataFrame({
        "Doc1": [27, 3, 0, 14],
        "Doc2": [4, 33, 33, 0],
        "Doc3": [24, 0, 29, 17]
    }, index=["car", "auto", "insurance", "best"])
    
    idf_series = pd.Series({
        "car": 1.65,
        "auto": 2.083,
        "insurance": 1.642,
        "best": 1.5
    })
    
    st.subheader("Term Frequency Table (TF)")
    st.write(tf_df)
    
    st.subheader("IDF Values")
    st.write(idf_series)
    
    tfidf_df = tf_df.multiply(idf_series, axis=0)
    st.subheader("TF-IDF Table")
    st.write(tfidf_df)
    
    st.subheader("Query Scoring")
    query = st.text_input("Enter query terms (space separated):", "car insurance")
    query_terms = query.lower().split()
    
    scores = {}
    for doc in tf_df.columns:
        score = sum(tfidf_df.loc[term, doc] for term in query_terms if term in tfidf_df.index)
        scores[doc] = score
    scores_df = pd.DataFrame.from_dict(scores, orient="index", columns=["Score"])
    st.write("Scores for query:", scores_df)
    
    if st.checkbox("Apply Euclidean Normalization to TF"):
        norm_df = tf_df.div(np.sqrt((tf_df**2).sum()), axis=1)
        st.write("Normalized TF Table:", norm_df)

# ------------------------
# Question 2: Cosine Similarity + Analogy Solver (Expanded TF table)
# ------------------------
elif option.startswith("Question 2"):
    st.header("Cosine Similarity & Word Analogies")
    
    # Expanded TF table
    tf_df = pd.DataFrame({
        "Doc1": [27, 3, 0, 14, 5, 9],
        "Doc2": [4, 33, 33, 0, 15, 6],
        "Doc3": [24, 0, 29, 17, 7, 12]
    }, index=["car", "auto", "insurance", "best", "policy", "claims"])
    
    st.subheader("Term Frequency Table (TF)")
    st.write(tf_df)
    
    cosine_matrix = cosine_similarity(tf_df.values)
    cosine_df = pd.DataFrame(cosine_matrix, index=tf_df.index, columns=tf_df.index)
    st.write("Cosine Similarity Matrix:", cosine_df)
    
    word = st.selectbox("Select a word to find nearest neighbours", tf_df.index)
    similarities = cosine_df[word].sort_values(ascending=False)
    st.write("Nearest neighbours:", similarities)
    
    st.subheader("Word Analogy Solver (A : B = C : X)")
    A = st.text_input("Enter word A", "car")
    B = st.text_input("Enter word B", "auto")
    C = st.text_input("Enter word C", "insurance")
    
    if A in tf_df.index and B in tf_df.index and C in tf_df.index:
        vec_A = tf_df.loc[A].values
        vec_B = tf_df.loc[B].values
        vec_C = tf_df.loc[C].values
        
        target_vec = vec_B - vec_A + vec_C
        
        similarities = {}
        for word in tf_df.index:
            if word not in [A, B, C]:
                sim = cosine_similarity([target_vec], [tf_df.loc[word].values])[0][0]
                similarities[word] = sim
        
        if similarities:
            best_match = max(similarities, key=similarities.get)
            st.write(f"**{A} : {B} = {C} : {best_match}** (Similarity = {similarities[best_match]:.4f})")
        else:
            st.warning("No candidate words available for analogy.")
    else:
        st.warning("One or more words not found in the vocabulary.")

# ------------------------
# Question 3: PMI (Preloaded counts)
# ------------------------
elif option.startswith("Question 3"):
    st.header("Pointwise Mutual Information (PMI) - From Preloaded Data")
    
    total_word_count = 233  # Total number of words in corpus
    
    # Word frequencies (occurrences)
    word_freq = {
        "car": 55,
        "auto": 36,
        "insurance": 62,
        "best": 31
    }
    
    # Co-occurrence counts
    co_occurrence = {
        ("car", "auto"): 12,
        ("car", "insurance"): 15,
        ("car", "best"): 8,
        ("auto", "insurance"): 10,
        ("auto", "best"): 4,
        ("insurance", "best"): 5
    }
    
    # Calculate PMI
    pmi_data = {}
    for (w1, w2), co_count in co_occurrence.items():
        p_w1 = word_freq[w1] / total_word_count
        p_w2 = word_freq[w2] / total_word_count
        p_w1w2 = co_count / total_word_count
        if p_w1w2 > 0:
            pmi = math.log2(p_w1w2 / (p_w1 * p_w2))
        else:
            pmi = float('-inf')  # Handle zero co-occurrence
        pmi_data[(w1, w2)] = pmi
    
    # Display results
    st.subheader("Word Frequencies")
    st.write(pd.DataFrame(list(word_freq.items()), columns=["Word", "Frequency"]))
    
    st.subheader("Co-occurrence Counts")
    st.write(pd.DataFrame([(w1, w2, count) for (w1, w2), count in co_occurrence.items()],
                          columns=["Word 1", "Word 2", "Count"]))
    
    st.subheader("Calculated PMI Values")
    pmi_df = pd.DataFrame(
        [(w1, w2, round(val, 4)) for (w1, w2), val in pmi_data.items()],
        columns=["Word 1", "Word 2", "PMI"]
    )
    st.dataframe(pmi_df, use_container_width=True)
