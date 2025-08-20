import streamlit as st
import re
import nltk
import pandas as pd
import math
from collections import defaultdict, OrderedDict
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.metrics.distance import edit_distance
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk import pos_tag

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

st.title("Lab4")

question = st.sidebar.selectbox("Choose a question", [
    "Question 1: Positional Index",
    "Question 2: Binary Word-Document Matrix",
    "Question 3: Frequency Index + Edit Distance",
    "Question 4: Levenshtein Edit Distance",
    "Question 5: POS Tagging using HMM",
    "Question 6: Word Sense Disambiguation (Lesk)"
])

# -----------------------------------
if question == "Question 1: Positional Index":
    st.header("Question 1: Positional Index")

    docs = {
        1: "I am a student, and I currently take MDS472C. I was a student in MDS331 last trimester.",
        2: "I was a student. I have taken MDS472C."
    }

    # Display the documents
    st.subheader("Input Documents")
    for doc_id, content in docs.items():
        st.write(f"**Doc {doc_id}:** {content}")

    # Preprocessing
    tokenized_docs = {}
    for doc_id, text in docs.items():
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        tokenized_docs[doc_id] = tokens

    # Build positional index
    positional_index = defaultdict(lambda: defaultdict(list))
    for doc_id, tokens in tokenized_docs.items():
        for pos, word in enumerate(tokens):
            positional_index[word][doc_id].append(pos)

    # Display full positional index
    st.subheader("Full Positional Index")
    for word in sorted(positional_index):
        st.text(f"{word}: {dict(positional_index[word])}")

    # User input for search
    words = st.text_input("Enter words to search (space separated)", "student MDS472C").lower().split()
    for word in words:
        if word in positional_index:
            st.text(f"Positions of '{word}': {dict(positional_index[word])}")
        else:
            st.text(f"'{word}' not found.")

# -----------------------------------
elif question == "Question 2: Binary Word-Document Matrix":
    st.header("Question 2: Binary Word-Document Matrix")
    docs = {
        'Doc1': "I am a student, and I currently take MDS472C. I was a student in MDS331 last trimester.",
        'Doc2': "I was a student. I have taken MDS472C."
    }

    tokenized_docs = {}
    for doc_id, text in docs.items():
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        tokenized_docs[doc_id] = tokens

    vocab = sorted(set(word for tokens in tokenized_docs.values() for word in tokens))
    binary_matrix = {
        doc_id: [1 if word in tokens else 0 for word in vocab]
        for doc_id, tokens in tokenized_docs.items()
    }

    df = pd.DataFrame.from_dict(binary_matrix, orient='index', columns=vocab)
    st.dataframe(df)

# -----------------------------------
elif question == "Question 3: Frequency Index + Edit Distance":
    st.header("Question 3: Frequency Index + Edit Distance")
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    num_docs = st.number_input("Enter number of documents", min_value=2, max_value=10, value=2)
    docs = [st.text_area(f"Enter Document {i+1}") for i in range(num_docs)]

    preprocessed_docs = []
    all_tokens_sequence = []
    word_variants = OrderedDict()

    for doc in docs:
        doc = doc.lower()
        doc = re.sub(r'[^\w\s]', '', doc)
        tokens = word_tokenize(doc)
        for word in tokens:
            stemmed = ps.stem(word)
            lemmatized = lemmatizer.lemmatize(word)
            word_variants[word] = (stemmed, lemmatized)
        lemmatized_list = [lemmatizer.lemmatize(word) for word in tokens]
        all_tokens_sequence.extend(lemmatized_list)
        preprocessed_docs.append(lemmatized_list)

    word_freq = defaultdict(int)
    for doc_tokens in preprocessed_docs:
        for word in doc_tokens:
            word_freq[word] += 1

    sort_option = st.selectbox("Sort by", ["frequency", "sequence"])
    if sort_option == 'frequency':
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    else:
        seen = set()
        ordered_words = OrderedDict()
        for word in all_tokens_sequence:
            if word not in seen:
                seen.add(word)
                ordered_words[word] = word_freq[word]
        sorted_words = list(ordered_words.items())

    # Build table for DataFrame display
    table_data = []
    added_words = set()

    for word, freq in sorted_words:
        for orig_word, (stem, lemma) in word_variants.items():
            if lemmatizer.lemmatize(orig_word) == word and orig_word not in added_words:
                table_data.append({
                    'Word': orig_word,
                    'Stemmed': stem,
                    'Lemmatized': lemma,
                    'Frequency': freq
                })
                added_words.add(orig_word)
                break

    df_freq = pd.DataFrame(table_data)
    st.subheader("Sorted Word Frequency Table")
    st.dataframe(df_freq)

    word1 = st.text_input("Word 1 for Edit Distance")
    word2 = st.text_input("Word 2 for Edit Distance")
    if word1 and word2:
        distance = edit_distance(word1.lower(), word2.lower())
        st.write(f"Edit distance between '{word1}' and '{word2}': {distance}")


# -----------------------------------
elif question == "Question 4: Levenshtein Edit Distance":
    st.header("Question 4: Levenshtein Edit Distance")

    wordA = st.text_input("Word A", "characterization")
    wordB = st.text_input("Word B", "categorization")

    def levenshtein_edit_distance(wordA, wordB):
        m, n = len(wordA), len(wordB)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        op = [[''] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
            op[i][0] = 'D'
        for j in range(n + 1):
            dp[0][j] = j
            op[0][j] = 'I'
        op[0][0] = ' '

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if wordA[i - 1] == wordB[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                    op[i][j] = 'M'
                else:
                    choices = [
                        (dp[i - 1][j - 1] + 1, 'S'),
                        (dp[i][j - 1] + 1, 'I'),
                        (dp[i - 1][j] + 1, 'D')
                    ]
                    dp[i][j], op[i][j] = min(choices)

        aligned_A, aligned_B, operations = [], [], []
        i, j = m, n
        ins = dels = subs = matches = 0

        while i > 0 or j > 0:
            if i > 0 and j > 0 and op[i][j] == 'M':
                aligned_A.append(wordA[i - 1])
                aligned_B.append(wordB[j - 1])
                operations.append('*')
                matches += 1
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and op[i][j] == 'S':
                aligned_A.append(wordA[i - 1])
                aligned_B.append(wordB[j - 1])
                operations.append('s')
                subs += 1
                i -= 1
                j -= 1
            elif j > 0 and op[i][j] == 'I':
                aligned_A.append('-')
                aligned_B.append(wordB[j - 1])
                operations.append('i')
                ins += 1
                j -= 1
            elif i > 0 and op[i][j] == 'D':
                aligned_A.append(wordA[i - 1])
                aligned_B.append('-')
                operations.append('d')
                dels += 1
                i -= 1

        aligned_A = ''.join(aligned_A[::-1])
        aligned_B = ''.join(aligned_B[::-1])
        operations = ''.join(operations[::-1])

        return dp[m][n], aligned_A, aligned_B, operations, ins, dels, subs, matches

    if wordA and wordB:
        dist, a_align, b_align, ops, ins, dels, subs, matches = levenshtein_edit_distance(wordA, wordB)
        st.text(f"Word A : {a_align}")
        st.text(f"Word B : {b_align}")
        st.text(f"Opertn : {ops}")
        st.write(f"Edit Distance: {dist}")
        st.write(f"Insertions: {ins}, Deletions: {dels}, Substitutions: {subs}, Matches: {matches}")

# -----------------------------------
elif question == "Question 5: POS Tagging using HMM":
    st.header("Question 5: HMM POS Tagging")

    tagged_sentences = [
        [('The', 'DET'), ('cat', 'NOUN'), ('chased', 'VERB'), ('the', 'DET'), ('rat', 'NOUN')],
        [('A', 'DET'), ('rat', 'NOUN'), ('can', 'AUX'), ('run', 'VERB')],
        [('The', 'DET'), ('dog', 'NOUN'), ('can', 'AUX'), ('chase', 'VERB'), ('the', 'DET'), ('cat', 'NOUN')]
    ]

    transition_counts = defaultdict(lambda: defaultdict(int))
    emission_counts = defaultdict(lambda: defaultdict(int))
    tag_counts = defaultdict(int)

    # Count transitions and emissions
    for sentence in tagged_sentences:
        prev_tag = 'START'
        for word, tag in sentence:
            word = word.lower()
            tag_counts[tag] += 1
            emission_counts[tag][word] += 1
            transition_counts[prev_tag][tag] += 1
            prev_tag = tag
        transition_counts[prev_tag]['END'] += 1

    # Calculate probabilities with add-one smoothing
    def calc_prob(count_dict):
        prob_dict = {}
        for prev, tags in count_dict.items():
            total = sum(tags.values()) + len(tags)  # denominator with smoothing
            prob_dict[prev] = {}
            for tag, count in tags.items():
                prob_dict[prev][tag] = (count + 1) / total
        return prob_dict

    transition_prob = calc_prob(transition_counts)
    emission_prob = calc_prob(emission_counts)
    tags = list(tag_counts.keys())
    tags_sorted = sorted(tags)

    # Prepare Transition Probability Matrix DataFrame
    all_prev_tags = sorted(set(transition_prob.keys()))
    all_next_tags = sorted({tag for d in transition_prob.values() for tag in d.keys()})

    trans_matrix = []
    for prev in all_prev_tags:
        row = []
        for nxt in all_next_tags:
            row.append(transition_prob.get(prev, {}).get(nxt, 0))
        trans_matrix.append(row)

    df_transition = pd.DataFrame(trans_matrix, index=all_prev_tags, columns=all_next_tags)
    st.subheader("Transition Probability Matrix")
    st.dataframe(df_transition.style.format("{:.4f}"))

    # Prepare Emission Probability Matrix DataFrame
    all_words = sorted({word for tag in emission_prob for word in emission_prob[tag].keys()})
    emission_matrix = []
    for tag in tags_sorted:
        row = []
        for word in all_words:
            row.append(emission_prob.get(tag, {}).get(word, 0))
        emission_matrix.append(row)

    df_emission = pd.DataFrame(emission_matrix, index=tags_sorted, columns=all_words)
    st.subheader("Emission Probability Matrix")
    st.dataframe(df_emission.style.format("{:.4f}"))

    sentence = st.text_input("Enter sentence to tag", "The rat can chase the cat").lower().split()

    def viterbi(sentence):
        V = [{}]
        path = {}

        # Initialize base cases (t == 0)
        for tag in tags:
            trans_p = transition_prob.get('START', {}).get(tag, 1e-8)
            emit_p = emission_prob.get(tag, {}).get(sentence[0], 1e-8)
            V[0][tag] = math.log(trans_p) + math.log(emit_p)
            path[tag] = [tag]

        # Run Viterbi for t > 0
        for t in range(1, len(sentence)):
            V.append({})
            new_path = {}

            for curr_tag in tags:
                max_prob = float('-inf')
                best_prev_tag = None
                emit_p = emission_prob.get(curr_tag, {}).get(sentence[t], 1e-8)

                for prev_tag in tags:
                    trans_p = transition_prob.get(prev_tag, {}).get(curr_tag, 1e-8)
                    prob = V[t-1][prev_tag] + math.log(trans_p) + math.log(emit_p)
                    if prob > max_prob:
                        max_prob = prob
                        best_prev_tag = prev_tag

                V[t][curr_tag] = max_prob
                new_path[curr_tag] = path[best_prev_tag] + [curr_tag]

            path = new_path

        # Termination step
        max_final_prob = float('-inf')
        best_final_tag = None
        for tag in tags:
            trans_p = transition_prob.get(tag, {}).get('END', 1e-8)
            prob = V[-1][tag] + math.log(trans_p)
            if prob > max_final_prob:
                max_final_prob = prob
                best_final_tag = tag

        return path[best_final_tag]

    if sentence:
        tagged_sequence = viterbi(sentence)
        st.subheader("Tagged Sentence")

        # Nicely formatted inline tagged sentence: word/POS
        formatted_output = " ".join(f"{word}/{tag}" for word, tag in zip(sentence, tagged_sequence))
        st.markdown(f"**{formatted_output}**")

        # Optional table display (commented)
        # df_tagged = pd.DataFrame({"Word": sentence, "POS Tag": tagged_sequence})
        # st.table(df_tagged)
# -----------------------------------
elif question == "Question 6: Word Sense Disambiguation (Lesk)":
    st.header("Question 6: Word Sense Disambiguation (Lesk)")

    corpus = [
        "The jury reached a verdict after closing arguments.",
        "The bank allowed the customer to withdraw cash.",
        "A storm is expected to delay flights at the airport."
    ]

    def get_wordnet_pos(treebank_tag):
        return {'J': wn.ADJ, 'V': wn.VERB, 'N': wn.NOUN, 'R': wn.ADV}.get(treebank_tag[0])

    for sent in corpus:
        st.write(f"**Sentence:** {sent}")
        tokens = word_tokenize(sent)
        pos_tags = pos_tag(tokens)
        for word, pos in pos_tags:
            wn_pos = get_wordnet_pos(pos)
            if wn_pos:
                senses = wn.synsets(word, pos=wn_pos)
                st.write(f"**Word:** {word} ({pos}) — Senses: {len(senses)}")
                for s in senses:
                    st.write(f"- {s.name()}: {s.definition()}")
                best = lesk(tokens, word, pos=wn_pos)
                if best:
                    st.write(f"→ Lesk-chosen sense: {best.name()} → {best.definition()}")
                else:
                    st.write("→ No Lesk sense chosen.")

        st.markdown("---")
