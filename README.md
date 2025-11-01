# Natural Langage Processing: Mental Health Sentiment Analysis
## Problem
Social media posts can reveal important information about a person's mental state. Early detection of mental health issues is the first step towards getting the necessary treatment and support. The goal of this project is to develop an AI model that can detect signs of mental health distress, like anxiety, depression, or stress, from text. 

---

## Data Source
[Reddit Mental Health Data](https://www.kaggle.com/datasets/neelghoshal/reddit-mental-health-data)  
  
This dataset is sourced from subreddits concerned with mental health based conversations. The mental health concerns included in the dataset have the following mappings:  
  - 0 = Stress  
  - 1 = Depression
  - 2 = Bipolar disorder  
  - 3 = Personality disorder  
  - 4 = Anxiety  
  
This dataset is valuable because it provides real world examples of people expressing emotions and struggles in their own words which allows the models to learn the words associated with mental health distress.

---

## Review of Relevant Previous Efforts and Literature  
There are several previous projects that focus on applying NLP to mental illness detection. [This](https://pmc.ncbi.nlm.nih.gov/articles/PMC8993841/) similar project focuses on comparing both traditonal deep NLP methods for early detection of mental health issues. 

**My Contribution:**  
My project is unique because of its focus on social media text posts as the data source

---

## Model Evaluation Process & Metric Selection   
- **Metrics:**  
  - Accuracy  
  - Precision, Recall
  - F1-score  
  - ROC AUC  
- **Data Splits:** Stratified 80%/10%/10% split for train/validation/test 

All three approaches (naive, classical ML, and deep learning) are trained and evaluated on the same split. The results are compared directly against the naive baseline.

---

## Modeling Approach  
1. **Naive Baseline:** Classify texts as expressing mental health distress if they contain predefined keywords
2. **Classical ML Approach:**  Train a machine learning classifier on TF-IDF features of the text
3. **Deep Learning Approach:**  Fine-tune a BERT transformer model
### Data Processing Pipeline  
The raw dataset is a CSV with columns text, title, and target, where target indicates the mental health condition (0 = Stress, 1 = Depression, 2 = Bipolar disorder, 3 = Personality disorder, 4 = Anxiety). To avoid data leakage and class imbalancem, the posts are split by random sampling into train (80%), validation (10%), and test (10%) sets.

The text data is cleaned and preprocessed by lowercasing all text, removing URLs, special characters, extra whitespace, and stopwords. 

The CSVs are saved under the following folder structure under `data/processed/`:
```
data/processed/
├── cleaned_data.csv
├── processed/
├──── train.csv
├──── val.csv
└──── test.csv
```

### Models Evaluated and Model Selected  
- **Evaluated Models:**

| Model           | Accuracy | Precision | Recall | F1-score | Notes |
|-----------------|---------|-----------|--------|----------|-------|
| Naive Baseline  | 0.38    | 0.53      | 0.32   | 0.35     | Performs poorly on classes with few or no keywords|
| Classical ML    | 0.74  | 0.75      | 0.74   | 0.75     | Significant improvement over naive baseline |
| Deep Learning   | 0.77     | 0.78       | 0.77    | 0.77      | BERT, Most overall balanced model  |

- **Model Selected:** Deep Learning
- **Model Weights [here](https://huggingface.co/moosejuice13/nlp-mental-health-bert)**  

### Comparison to Naive Approach  
The deep learning BERT model significantly outperforms the naive baseline. It improves the accuracy from 0.38 to 0.77. This improvement in accuracy shows that the BERT model is better at capturing the contextual meaning of the text compared to simple keyword matching.

---

## Visual Interface Demo

<img width="1029" height="590" alt="Screenshot 2025-10-31 at 3 52 59 PM 1" src="https://github.com/user-attachments/assets/52e37b03-c007-4bc9-a1ed-99b57f2f39fd" />

Video demo of the project can be found [here](https://drive.google.com/file/d/1PnfSZxMuAPDfx_DPDCzTmzl8rY07m6AB/view?usp=sharing)  
Streamlit site can be found [here](https://nlp-mental-health-app-963698787646.us-central1.run.app/)

---

## Results and Conclusions  
The deep learning BERT model achieved an accuracy of 77% which outperformed both the naive baseline, which has an accuracy of 38% and the classical machine learning model, which has an accuracy of 74%. The model's ability to understand the contextual and semantic nuances in text allowed it to generalize better across different types of mental health expressions. This improvement demonstrates the transformers abilities to capture emotional cues that a simple keyword based search misses. 

While the model is not perfect, it does have potential to function as a support tool to monitor or analyze mental health discussions online. 

---

## Ethics Statement  

This project is intended for educational and research purposes only and should not be used to assess or diagnose mental health conditions. The model's outputs are not a professional diagnosis and should not replace professional care or clinical advice. Please seek medical attention if you are experiencing symptoms of a mental health condition. Any real-world application of this tool should undergo ethical review and validation to ensure accuracy, user safety, and data privacy.

---

## Instructions on How to Run the Code

1. Clone the Repository  
`git clone https://github.com/Mkhan13/nlp_mental_health_sentiment_analysis.git`  
`cd nlp_mental_health_sentiment_analysis`

3. Install Dependencies  
`pip install -r requirements.txt`

4. Run the Streamlit App  
`streamlit run main.py`  
The app will open in your browser  

6. Type or copy and paste text into the text box. Click predict to analyze the text sentiment.
