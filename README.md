# Natural Langage Processing: Anatomy Textbook Summarization
## Problem
Medical textbooks are dense and terminology-heavy which can make it difficult for students to quickly understand and retain key information. Anatomy, which is considered one of the hardest classes in medical school, requires detailed reading and repetition which can be time-consuming and overwhelming. This project uses natural language processing (NLP) to automatically summarize anatomy content into clear and concise summaries that are easier for students to review and understand.

---

## Data Source
**[InfoBooks Anatomy Ebook Conversions](https://www.kaggle.com/datasets/evidence/infobooks-anatomy-ebook-conversions)**  
This dataset contains PDFs of the following anatomy textbooks:
  1) Anatomy & Physiology. Volume 1 of 3
  2) Anatomy & Physiology. Volume 2 of 3
  3) Anatomy & Physiology. Volume 3 of 3
  4) Surgical Anatomy -
  Joseph Maclise
  5) Anatomy -
  Guus van der Bie
  6) Myology -
  Pearson - Higher Education
  7) Cardiovascular System -
  Pearson - Higher Education
  8) The Respiratory System -
  Pearson - Higher Education
  9) The Nervous System -
  Jim Swan
  10) The Muscular System -
  Jim Swan
  11) Human Anatomy Lecture Notes -
  Rebecca Bailey
  12) Anatomy of the Athlete -
  Colby College
  13) Bones of the Skull -
  Department of Oral and Maxillofacial Surgery
  14) The Amazing Human Body (Presentation) -
  Howie Baum
  15) Cranial skeleton (Neurocranium) (Presentation) -
  National Taiwan University
  16) The Skeletal System (Presentation) -
  Howie Baum
  17) Arthrology (Presentation) -
  Anatomy plcnet
  18) Bones and Joints (Presentation) -
  Leslie G. Dodd
  19) Medical Gross Anatomy Introduction (Presentation) -
  Nicholas Lutfi
  20) The Endocrine System (Presentation) -
  Howie Baum
  21) Spinal Cord, Spinal Nerves and Somatic Reflexes (Presentation) -
  Fisiokinesiterapia
  22) Functional anatomy: Female Genital System (Presentation) -
  Ie-Ming Shih
  23) Anatomy and Physiology of the Skin -
  Paul A.J. Kolarsick, Maria Ann Kolarsick, Carolyn Goodwin
  24) Introduction to Basic Human Anatomy (Presentation) -
  The Brookside Associates

---

## Review of Relevant Previous Efforts and Literature  
[This study](https://www.sciencedirect.com/science/article/pii/S0010482525001581) focuses on using NLP techniques to analyze electronic medical records. [This study](https://ieeexplore.ieee.org/document/9984491) uses NLP to summarize and simplify medical articles. [This collection](https://nlp.johnsnowlabs.com/medical_text_summarization) of live demos demonstrate how NLP can be used to summarize clinical texts.

**My Contribution:**  
While projects applying NLP to summarize medical texts have been conducted previously, my project focuses specifically on anatomy textbooks due to their especially difficult verbage.

---

## Model Evaluation Process & Metric Selection   
- **Metrics:**  
  - ROUGE
  - BERTScore
- **Data Splits:** Because this is a summarization task, the training data is 100% of the dataset 

All three approaches (naive, classical ML, and deep learning) are trained and evaluated on the same split. The results are compared directly against the naive baseline to quantify performance improvements.

---

## Modeling Approach  
1. **Naive Baseline:** Return the first few sentences of the passage as a summary
2. **Classical ML Approach:**  Use TextRank to select key sentences from the passage
3. **Deep Learning Approach:**  Train a BART model end-to-end to rephrase the content into comprehensive summaries

### Data Processing Pipeline  
The raw dataset consists of anatomy textbook passages stored as text files. To prepare the data for modeling, the text is cleaned to remove headers, footers, page numbers, and any non-text elements. Then the dataset is converted to text embeddings and stored into a searchable index for retreival later. 

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
