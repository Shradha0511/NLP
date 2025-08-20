import streamlit as st
import nltk
from nltk import word_tokenize, bigrams
from nltk.corpus import treebank
from collections import Counter, defaultdict
import math
import spacy

# === Download required data ===
nltk.download('punkt')
nltk.download('treebank')

# === Section 1: Bigram Language Model ===
corpus_text = """
I am Sam
I like green grapes.
John likes red apples.
He eats red apples.
They like green vegetables.
I eat fruits every day.
We all like fresh food.
Green grapes are sweet.
"""

def preprocess(text):
    tokens = word_tokenize(text.lower())
    return ['<s>'] + tokens + ['</s>']

tokens = []
for line in corpus_text.strip().split('\n'):
    tokens += preprocess(line)

unigram_counts = Counter(tokens)
bigram_counts = Counter(bigrams(tokens))
V = len(unigram_counts)

bigram_prob = defaultdict(float)
for (w1, w2), count in bigram_counts.items():
    bigram_prob[(w1, w2)] = count / unigram_counts[w1]

def compute_perplexity(test_tokens):
    N = len(test_tokens) - 1
    if N == 0:
        return float('inf')
    log_prob_sum = 0.0
    for i in range(1, len(test_tokens)):
        w1, w2 = test_tokens[i - 1], test_tokens[i]
        prob = (bigram_counts[(w1, w2)] + 1) / (unigram_counts[w1] + V)
        log_prob_sum += math.log(prob)
    return math.exp(-log_prob_sum / N)

# === Section 2: HMM POS Tagger (Enhanced) ===
transition_counts = defaultdict(Counter)
emission_counts = defaultdict(Counter)
tag_counts = Counter()

tagged_sents = treebank.tagged_sents()
for sent in tagged_sents:
    prev_tag = "<s>"
    tag_counts[prev_tag] += 1
    for word, tag in sent:
        word_lower = word.lower()
        transition_counts[prev_tag][tag] += 1
        emission_counts[tag][word_lower] += 1
        tag_counts[tag] += 1
        prev_tag = tag
    transition_counts[prev_tag]["</s>"] += 1
    tag_counts["</s>"] += 1

# Remove invalid tags
invalid_tags = {"<s>", "</s>", "-NONE-"}
states = [tag for tag in tag_counts if tag not in invalid_tags]

# Emission probability with OOV handling
def get_emission_prob(tag, word):
    w_lower = word.lower()

    # Seen word
    if w_lower in emission_counts[tag]:
        return emission_counts[tag][w_lower] / tag_counts[tag]

    # Determiners
    if w_lower in {"the", "a", "an"} and tag == "DT":
        return 0.05

    # Pronouns
    if w_lower in {"he", "she", "it", "they", "we", "i", "you"} and tag.startswith("PRP"):
        return 0.05

    # Proper nouns: names or capitalized words
    if word[0].isupper():
        return 0.01 if tag == "NNP" else 1e-6

    # Adjectives for colors/qualities
    if w_lower.endswith("y") or w_lower in {"pink", "blue", "green", "red", "black", "white", "beautiful"}:
        return 0.005 if tag.startswith("JJ") else 1e-6

    # Default tiny probability
    return 1e-6

# Viterbi algorithm with enhanced emission handling
def hmm_viterbi(words):
    V = [{}]
    path = {}

    # Initialization
    for y in states:
        trans_p = (transition_counts["<s>"][y] + 1) / (tag_counts["<s>"] + len(states))
        emis_p = get_emission_prob(y, words[0])
        V[0][y] = math.log(trans_p) + math.log(emis_p)
        path[y] = [y]

    # Recursion
    for t in range(1, len(words)):
        V.append({})
        new_path = {}
        for y in states:
            (prob, state) = max(
                (V[t - 1][y0] +
                 math.log((transition_counts[y0][y] + 1) / (tag_counts[y0] + len(states))) +
                 math.log(get_emission_prob(y, words[t])),
                 y0)
                for y0 in states
            )
            V[t][y] = prob
            new_path[y] = path[state] + [y]
        path = new_path

    # Termination
    (prob, final_state) = max((V[-1][y], y) for y in states)
    return path[final_state]

# === Section 3: BIO NER ===
nlp = spacy.load("en_core_web_sm")

def get_bio_ner_tags(text):
    doc = nlp(text)
    bio_tags = []
    for token in doc:
        ent = token.ent_iob_
        if ent == 'O':
            bio_tags.append('O')
        else:
            bio_tags.append(f"{ent}-{token.ent_type_}")
    return [token.text for token in doc], bio_tags

# === Streamlit App ===
st.title("NLP Toolkit App")

# Bigram Language Model UI
st.header("Bigram Language Model")
input_sentence = st.text_input("Enter a sentence to calculate perplexity:")
if input_sentence:
    test_tokens = preprocess(input_sentence)
    pp = compute_perplexity(test_tokens)
    st.write(f"Perplexity: {pp:.4f}")

w1 = st.text_input("Enter first word for bigram probability:")
w2 = st.text_input("Enter second word:")
if w1 and w2:
    prob = bigram_prob.get((w1.lower(), w2.lower()), 0.0)
    st.write(f"P({w2}/{w1}) = {prob:.6f}")

# HMM POS Tagger UI
st.header("HMM POS Tagger")
pos_sentence = st.text_input("Enter a sentence for POS tagging:")
if pos_sentence:
    tokens = nltk.word_tokenize(pos_sentence)
    tags = hmm_viterbi(tokens)
    st.table({"Token": tokens, "Tag": tags})

# BIO NER UI
st.header("BIO Named Entity Recognition")
ner_sentence = st.text_input("Enter a sentence for BIO NER:")
if ner_sentence:
    tokens, tags = get_bio_ner_tags(ner_sentence)
    st.table({"Token": tokens, "BIO Tag": tags})

