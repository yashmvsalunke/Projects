import os
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Function to extract article text from a given URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract article title
        title = soup.title.text.strip()
        
        # Extract article text within div.td-post-content.tagdiv-type
        content_div = soup.find('div', class_='td-post-content tagdiv-type')
        if content_div:
            paragraphs = content_div.find_all('p')
            article_text = '\n'.join([p.text.strip() for p in paragraphs])
        else:
            article_text = "No data found in the specified div tag."
        
        return title, article_text
    except Exception as e:
        print(f"Error extracting data from {url}: {e}")
        return None, None

# Load input data from Excel file
input_data = pd.read_excel('Input.xlsx')

# Create a folder for storing extracted data
output_folder = 'data_extracted'
os.makedirs(output_folder, exist_ok=True)

# Iterate through each row in the input data
for index, row in input_data.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    
    # Extract text from the URL
    title, article_text = extract_text_from_url(url)
    
    if title is not None and article_text is not None:
        # Save the extracted text in a text file within the 'data_extracted' folder
        output_path = os.path.join(output_folder, f'{url_id}.txt')
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(f'{title}\n\n{article_text}')

print("Extraction completed.")

import os
import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords


# Directory containing StopWords files
stopwords_directory = 'C:\\Users\\MAHESH\\Downloads\\2\\StopWords'

# Output combined StopWords file
output_stopwords_file = 'C:\\Users\\MAHESH\\Downloads\\2\\StopWords.txt'

# List to store combined stopwords
combined_stopwords = set()

# Iterate through each file in the StopWords directory
for filename in os.listdir(stopwords_directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(stopwords_directory, filename)
        
        # Read stopwords from the current file
        with open(file_path, 'r', encoding='latin-1') as current_file:
            stopwords_content = current_file.read().splitlines()
            
            # Add stopwords to the combined list
            combined_stopwords.update(stopwords_content)

# Write the combined stopwords to the output file
with open(output_stopwords_file, 'w', encoding='utf-8') as output_file:
    output_file.write('\n'.join(combined_stopwords))

print("Combined StopWords file created:", output_stopwords_file)

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load stop words from the combined StopWords.txt file
stopwords_path = 'C:\\Users\\MAHESH\\Downloads\\2\\StopWords.txt'
stop_words = set()
with open(stopwords_path, 'r', encoding='utf-8') as stop_words_file:
    stop_words = set(stop_words_file.read().splitlines())

# Load positive and negative words from the MasterDictionary folder
positive_words_path = 'C:\\Users\\MAHESH\\Downloads\\2\\MasterDictionary\\positive-words.txt'
negative_words_path = 'C:\\Users\\MAHESH\\Downloads\\2\\MasterDictionary\\negative-words.txt'

positive_words = set()
with open(positive_words_path, 'r', encoding='utf-8') as positive_words_file:
    positive_words = set(positive_words_file.read().splitlines())

negative_words = set()
with open(negative_words_path, 'r', encoding='utf-8', errors='ignore') as negative_words_file:
    negative_words = set(negative_words_file.read().splitlines())

# Function to perform sentiment analysis
def sentiment_analysis(text):
    cleaned_words = [word.lower() for word in word_tokenize(text) if word.isalpha() and word.lower() not in stop_words]
    
    positive_score = sum(1 for word in cleaned_words if word in positive_words)
    negative_score = sum(1 for word in cleaned_words if word in negative_words)

    # Calculate polarity score
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)

    # Calculate subjectivity score
    subjectivity_score = (positive_score + negative_score) / (len(cleaned_words) + 0.000001)

    return positive_score, negative_score, polarity_score, subjectivity_score

# Function to perform readability analysis
def readability_analysis(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    # Calculate average sentence length
    average_sentence_length = len(words) / len(sentences)

    # Calculate percentage of complex words
    complex_words = [word for word in words if syllable_count(word) > 2]
    percentage_complex_words = len(complex_words) / len(words)

    # Calculate Fog Index
    fog_index = 0.4 * (average_sentence_length + percentage_complex_words)

    # Calculate average number of words per sentence
    average_words_per_sentence = len(words) / len(sentences)

    # Calculate complex word count
    complex_word_count = len(complex_words)

    # Calculate word count
    word_count = len(words)

    # Calculate syllable count per word
    syllable_count_per_word = sum(syllable_count(word) for word in words) / len(words)

    # Calculate personal pronoun count
    personal_pronouns_count = len(re.findall(r'\b(?:I|we|my|ours|us)\b', text, flags=re.IGNORECASE))

    # Calculate average word length
    average_word_length = sum(len(word) for word in words) / len(words)

    return (
        average_sentence_length,
        percentage_complex_words,
        fog_index,
        average_words_per_sentence,
        complex_word_count,
        word_count,
        syllable_count_per_word,
        personal_pronouns_count,
        average_word_length
    )

# Function to count syllables in a word
def syllable_count(word):
    vowels = "aeiouy"
    count = 0
    word = word.lower()
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith('e'):
        count -= 1
    if count == 0:
        count += 1
    return count

# Load input data from Excel file
input_data = pd.read_excel('Input.xlsx')

# Create a list to store analysis results
analysis_results = []

# Iterate through each row in the input data
for index, row in input_data.iterrows():
    url_id = row['URL_ID']
    text_file_path = os.path.join('data_extracted', f'{url_id}.txt')

    # Read text from the file
    with open(text_file_path, 'r', encoding='utf-8') as text_file:
        text = text_file.read()

    # Perform analysis
    positive_score, negative_score, polarity_score, subjectivity_score = sentiment_analysis(text)
    readability_results = readability_analysis(text)

    # Combine the results
    results = [
        url_id,
        positive_score,
        negative_score,
        polarity_score,
        subjectivity_score,
        *readability_results
    ]

    analysis_results.append(results)

# Define the output CSV file
output_csv_file = 'analysis_results.csv'

# Update the number of columns in the columns list
columns = [
    "URL_ID",
    "positive_score",
    "negative_score",
    "polarity_score",
    "subjectivity_score",
    "average_sentence_length",
    "percentage_complex_words",
    "fog_index",
    "average_words_per_sentence",
    "complex_word_count",
    "word_count",
    "syllable_count_per_word",
    "personal_pronouns_count",
    "average_word_length"
]

df = pd.DataFrame(analysis_results, columns=columns)
df.to_csv(output_csv_file, index=False)

print(f"Analysis results saved to {output_csv_file}")
