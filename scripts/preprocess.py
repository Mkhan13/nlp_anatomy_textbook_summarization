import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import os

# NLTK download
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# File paths
RAW_CSV = 'data/raw.csv'
PROCESSED_CSV = 'data/processed/cleaned_data.csv'
TRAIN_CSV = 'data/processed/train.csv'
VALIDATION_CSV = 'data/processed/validation.csv'
TEST_CSV = 'data/processed/test.csv'

os.makedirs(os.path.dirname(PROCESSED_CSV), exist_ok=True)

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    '''Function to clean text data by removing URLs, punctuation, stopwords, and lemmatizing'''
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r'http\S+|www\S+|/r/\S+|@\S+', ' ', text) # Remove URLs and Reddit handles
    text = re.sub(r'[-•]+', ' ', text) # Remove bullet points and multiple dashes
    text = re.sub(r'\n+', ' ', text)     # Replace newlines with spaces
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation

    tokens = text.split() # Tokenize

    clean_tokens = []
    for t in tokens:
        if t not in stop_words: # Remove stopwords
            clean_tokens.append(lemmatizer.lemmatize(t)) # Lemmatize

    return ' '.join(clean_tokens)

def main():
    df = pd.read_csv(RAW_CSV)
    df['text'] = df['text'].apply(clean_text)
    df.to_csv(PROCESSED_CSV)
    
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target']) # Train split
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['target']) # Test and Validation split

    train_df.to_csv(TRAIN_CSV)
    val_df.to_csv(VALIDATION_CSV)
    test_df.to_csv(TEST_CSV)

if __name__ == "__main__":
    main()