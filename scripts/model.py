import os
import torch
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)

def load_data():
    '''Load and clean train, validation, and test data'''
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/validation.csv')
    test_df = pd.read_csv('data/processed/test.csv')

    # Drop NaN values
    train_df = train_df.dropna(subset=['text', 'target'])
    val_df = val_df.dropna(subset=['text', 'target'])
    test_df = test_df.dropna(subset=['text', 'target'])

    return train_df, val_df, test_df

def prepare_datasets(train_df, val_df, test_df):
    '''Convert DataFrames to tokenized Hugging Face Datasets'''
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    for ds in [train_dataset, val_dataset, test_dataset]:
        ds.map(tokenize, batched=True)

    train_dataset = train_dataset.rename_column('target', 'labels')
    val_dataset = val_dataset.rename_column('target', 'labels')
    test_dataset = test_dataset.rename_column('target', 'labels')

    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    return train_dataset, val_dataset, test_dataset

def train_model(train_dataset, val_dataset, num_labels):
    '''Train and save model'''
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy='epoch',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    os.makedirs('./models', exist_ok=True)
    trainer.save_model('./models/bert_model')
    tokenizer.save_pretrained('./models/bert_tokenizer')

    return model, trainer

def evaluate_model(trainer, test_dataset, test_df):
    '''Evaluate model and print classification report.'''
    preds = trainer.predict(test_dataset)
    y_true = test_df['target']
    y_pred = preds.predictions.argmax(axis=1)
    print(classification_report(y_true, y_pred))

def push_to_hub():
    '''Push the trained model to Hugging Face Hub'''
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    login(token=hf_token)

    model = AutoModelForSequenceClassification.from_pretrained('./models/bert_model')
    tokenizer = AutoTokenizer.from_pretrained('./models/bert_tokenizer')
    repo_name = 'moosejuice13/nlp-mental-health-bert'

    model.push_to_hub(repo_name, use_auth_token=hf_token)
    tokenizer.push_to_hub(repo_name, use_auth_token=hf_token)
    print(f'https://huggingface.co/{repo_name}')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_df, val_df, test_df = load_data() # Load data
    train_dataset, val_dataset, test_dataset = prepare_datasets(train_df, val_df, test_df) # Prepare datasets
    num_labels = len(train_df['target'].unique()) # Determine number of unique labels

    model, trainer = train_model(train_dataset, val_dataset, num_labels) # Train the model
    evaluate_model(trainer, test_dataset, test_df) # Evaluate the model
    push_to_hub() # Push model to HuggingFace hub

if __name__ == '__main__':
    main()