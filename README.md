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
3. **Deep Learning Approach:**  Fine-tune a BERT transformer model to predict distress from the full text context.

### Data Processing Pipeline  
 

### Models Evaluated and Model Selected  
- **Evaluated Models:**

| Approach             | ROUGE | BERTScore | Notes |
|----------------------|-------|-----------|-------|
| **Naive Baseline**   |       |           |       |
| **Classical ML**     |       |           |       |
| **Deep Learning**    |       |           |       |


- **Model Selected:** 

### Comparison to Naive Approach  


---

## Visual Interface Demo

Video demo of the project can be found [here](LINK HERE)  
Streamlit site can be found [here](LINK HERE)

---

## Results and Conclusions  


---

## Ethics Statement  



---

## Instructions on How to Run the Code

1. Clone the Repository  
`git clone https://github.com/Mkhan13/nlp_anatomy_textbook_summarization.git`  
`cd nlp_anatomy_textbook_summarization`

3. Install Dependencies  
`pip install -r requirements.txt`

4. Run the Streamlit App  
`streamlit run main.py`  
The app will open in your browser  

6. {WHAT USER DOES TO APP} 
