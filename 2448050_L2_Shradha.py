# %% [markdown]
# ### MDS472C: NATURAL LANGUAGE PROCESSING
# ### Lab2: Edit Distance and Applications
# #### Date : 30 June 2025
# ### 2448050

# %% [markdown]
# ##### Q1. Edit distance (Manual) Solve exercise 2.4 and 2.5 from Text book of Speech and Language Processing of Daniel Jurafsky and team
# ##### 2.4 Compute the edit distance (using insertion cost 1, deletion cost 1, substitution cost 1) of “leda” to “deal”. Show your work (using the edit distance grid).
#  

# %%
from IPython.display import Image
Image(r"C:\Users\shrad\OneDrive\Documents\NLP\lab2(2).jpg")

# %% [markdown]
# ##### 2.5 Figure out whether the “drive” is closer to “brief” or to “divers” and what the edit distance is to each. You may use any version of distance that you like.

# %%
from IPython.display import Image
Image(r"C:\Users\shrad\OneDrive\Documents\NLP\lab2(1).jpg")

# %% [markdown]
# ##### Q2. Edit distance (Implementation)
# 
# ##### 2.6 Implement a minimum edit distance algorithm and use your hand-computed results to check your code.

# %%
def levenshtein_distance(str1, str2):
    m, n = len(str1), len(str2)
    
    # Create a (m+1) x (n+1) matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # initializing the firrt row and column.
    for i in range(m + 1):
        dp[i][0] = i  # Deletion
    for j in range(n + 1):
        dp[0][j] = j  # Insertion

    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No cost
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],    # Deletion
                    dp[i][j - 1],    # Insertion
                    dp[i - 1][j - 1] # Substitution
                )

    return dp[m][n]

str1 = "leda"
str2 = "deal"
distance = levenshtein_distance(str1, str2)
print(f"The edit distance between '{str1}' and '{str2}' is {distance}")


# %% [markdown]
# ##### Test Cases:

# %%
#test case 1:
str1 = "kitten"
str2 = "sitting"
distance = levenshtein_distance(str1, str2)
print(f"The edit distance between '{str1}' and '{str2}' is {distance}")

# %%
# test case 2:
str1 = "intention"
str2 = "execution"
print(f"Edit distance between '{str1}' and '{str2}' is {distance}")

# %% [markdown]
# ##### Program Description
# This program calculates the Levenshtein Distance between two strings, which is the minimum number of insertions, deletions, or substitutions needed to convert one string into another. It uses dynamic programming to fill a matrix where each cell represents the edit distance between substrings. The final value in the matrix gives the total minimum edits required.

# %% [markdown]
# ##### Program Logic Explanation
# 1. Two Strings as input
# There will be two input strings given by the user. In this case:
# 
# First string (original): "leda"
# 
# Second string (target): "deal"
# 
# 2. Create a Comparison Grid
# A matrix (grid) is created to track the number of steps needed to change parts of the original string into the target string.
# 
# The rows represent characters of the first string.
# 
# The columns represent characters of the second string.
# 
# 3. Initialize the Matrix
# The first row and the first column are filled with increasing numbers.
# 
# This represents the cost of converting an empty string to part of the other string using only insertions or deletions.
# 
# 4. Compare Characters One by One
# The program goes through each character of both strings and compares them:
# 
# If the characters are the same, no change is needed, so the value is copied from the diagonal cell.
# 
# If they are different, the program adds 1 to the minimum value among:
# 
# -Deletion (cell above)
# 
# -Insertion (cell to the left)
# 
# -Substitution (diagonal cell)
# 
# 5. Track minimum Edit Cost
# Each cell in the grid stores the least number of changes needed to match the corresponding parts of the strings.
# 
# 6. Final result in Bottom-Right Cell
# Once the entire table is filled, the bottom-right cell contains the minimum number of steps needed to change the full original string into the target string.
# 
# 7. Display the Result
# The program prints this final value, representing the minimum number of insertions, deletions, or substitutions needed.

# %% [markdown]
# ##### Q3. Implement Sequence Alignment
# ##### Write a program to align the given sequence of input text A and B
# 

# %%
def align_sequences(A, B):
    m, n = len(A), len(B)
    gap_penalty = -2
    match_score = 1
    mismatch_penalty = -1

    # Initialize score matrix
    score = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill first row and column
    for i in range(m + 1):
        score[i][0] = i * gap_penalty
    for j in range(n + 1):
        score[0][j] = j * gap_penalty

    # Fill the rest of the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if A[i - 1] == B[j - 1]:
                match = score[i - 1][j - 1] + match_score
            else:
                match = score[i - 1][j - 1] + mismatch_penalty
            delete = score[i - 1][j] + gap_penalty
            insert = score[i][j - 1] + gap_penalty
            score[i][j] = max(match, delete, insert)

    # Traceback to get alignment
    aligned_A = ""
    aligned_B = ""
    i, j = m, n
    while i > 0 and j > 0:
        current = score[i][j]
        diagonal = score[i - 1][j - 1]
        up = score[i - 1][j]
        left = score[i][j - 1]

        if A[i - 1] == B[j - 1] or current == diagonal + mismatch_penalty:
            aligned_A = A[i - 1] + aligned_A
            aligned_B = B[j - 1] + aligned_B
            i -= 1
            j -= 1
        elif current == up + gap_penalty:
            aligned_A = A[i - 1] + aligned_A
            aligned_B = "-" + aligned_B
            i -= 1
        else:
            aligned_A = "-" + aligned_A
            aligned_B = B[j - 1] + aligned_B
            j -= 1

    # Add remaining gaps if any
    while i > 0:
        aligned_A = A[i - 1] + aligned_A
        aligned_B = "-" + aligned_B
        i -= 1
    while j > 0:
        aligned_A = "-" + aligned_A
        aligned_B = B[j - 1] + aligned_B
        j -= 1

    return aligned_A, aligned_B

A = "AGGCTATCACCTGACCTCCAGGCCGATGCCC"
B = "TAGCTATCACGACCGCGGTCGATTTGCCCGAC"

aligned_A, aligned_B = align_sequences(A, B)
print("The non-Aligned one is :",aligned_A)
print("The aligned one is :",aligned_B)


# %% [markdown]
# ##### Test Cases:
# 

# %%
# test case 1:
A = "ACCGT"
B = "ACG"
aligned_A, aligned_B = align_sequences(A, B)
print("Aligned A:", aligned_A)
print("Aligned B:", aligned_B)


# %%
# test case 2:
A = "ACTG"
B = "ACGT"

aligned_A, aligned_B = align_sequences(A, B)
print("Aligned A:", aligned_A)
print("Aligned B:", aligned_B)


# %% [markdown]
# ##### Program Description
# This program aligns two input sequences (like DNA or protein strings) using the Needleman-Wunsch algorithm for global alignment.
# 
# It shows how similar two sequences are by matching characters, inserting gaps (-), or replacing mismatches — so that the overall alignment score is maximized.

# %% [markdown]
# ##### Logic Explanation (Step by Step)
# 1. Define Scoring System
# Match: +1
# 
# Mismatch: -1
# 
# Gap (insertion or deletion): -2
# These scores are used to compute the best alignment.
# 
# 2. Set Up a Score Matrix
# A grid (or table) is created to compare all parts of string A vs string B.
# 
# Each cell in the grid represents the best score you can get by aligning parts of A and B up to that point.
# 
# 3. Initialize First Row and Column
# The first row and column are filled with cumulative gap penalties.
# 
# This assumes that aligning a string with an empty string requires a number of insertions or deletions equal to its length.
# 
# 4. Fill the Rest of the Table
# For each cell:
# 
# If the characters match, the score is copied from the diagonal cell plus 1.
# 
# If not, the score is the best (maximum) of:
# 
# Diagonal cell + mismatch penalty
# 
# Cell above + gap penalty (deletion)
# 
# Cell to the left + gap penalty (insertion)
# 
# This ensures we build up the highest possible score for each partial alignment.
# 
# 5. Trace Back to Get the Alignment
# Start from the bottom-right cell of the grid.
# 
# Move in reverse to find the path that gave the best score:
# 
# If coming from diagonal, align the two characters.
# 
# If coming from above, align character from A with a gap.
# 
# If coming from left, align character from B with a gap.
# 
# This reconstructs the aligned sequences.
# 
# 6. Handle Remaining Characters
# If one string finishes before the other, the rest of the characters are aligned with gaps.
# 
# 7. Output
# The function returns the two new aligned strings with - characters added wherever needed.
# 
# The print statement shows both aligned strings.

# %% [markdown]
# ##### Q4. Application to correct the  misspelled words. (optional)

# %%
def edit_distance(str1, str2):
    n = len(str1)
    m = len(str2)
    dp = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Deletion
                    dp[i][j - 1],      # Insertion
                    dp[i - 1][j - 1]   # Substitution
                )
    return dp[n][m]
def correct_word(misspelled_word, dictionary):
    min_dist = float('inf')
    correction = ""
    
    for word in dictionary:
        dist = edit_distance(misspelled_word, word)
        if dist < min_dist:
            min_dist = dist
            correction = word        
    return correction
# Test Case
dictionary = ["spelling", "spilling", "splint", "speaking"]
misspelled_word = "speling"

suggestion = correct_word(misspelled_word, dictionary)
print(f"Suggested correction for '{misspelled_word}': {suggestion}")


# %% [markdown]
# ##### Test Cases:

# %%
dictionary = ["receive", "deceive", "relieve", "review"]
misspelled_word = "recieve"

suggestion = correct_word(misspelled_word, dictionary)
print(f"Suggested correction for '{misspelled_word}': {suggestion}")

# %%
dictionary = ["accommodation", "recommendation", "communication", "commemoration"]
misspelled_word = "acommodation"

suggestion = correct_word(misspelled_word, dictionary)
print(f"Suggested correction for '{misspelled_word}': {suggestion}")

# %% [markdown]
# ##### Program Description
# The program takes a misspelled word as input and suggests the most likely correct word by comparing it to a list of dictionary words. It uses the concept of minimum edit distance (Levenshtein distance) to measure how similar the misspelled word is to each word in the dictionary. The word with the smallest edit distance is suggested as the correction.
# 

# %% [markdown]
# ##### Program Logic
# The program uses a function to compute the minimum edit distance between two words.
# 
# It compares the misspelled word to each word in the dictionary using this function.
# 
# The edit distance counts the minimum number of insertions, deletions, and substitutions required to convert one word into another.
# 
# The word from the dictionary with the smallest edit distance is selected as the suggested correction.


