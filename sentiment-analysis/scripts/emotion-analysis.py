import os
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification
from tqdm import tqdm

# Importing the csv dataset
import_dataset = '../sentiment-model/test-posts-scores.csv'
df_data = pd.read_csv(import_dataset)

# Preparing the dataset for the network this includes tokenization, encoding and creating dataloaders.
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

encodings = tokenizer(df_data['Text'].tolist(), truncation=True, padding=True, return_tensors='pt')

dataset = torch.utils.data.TensorDataset(
    encodings['input_ids'], 
    encodings['attention_mask']
)
dataloader = DataLoader(dataset, batch_size=6, shuffle=False)

# Loading the model
model= torch.load('bert-go-emotion.pth')

# Using the model to perform emotion analysis on the dataset
model.eval()
list_predicted_scores = {'V': [], 'A': [], 'D': []}

for batch in tqdm(dataloader):
    with torch.no_grad():
        input_ids, attention_mask = batch

        # Obtaining the emotion scores
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        predicted_scores = output.logits

        # Writing the emotion scores to the list
        list_predicted_scores['V'].extend(predicted_scores[:, 0].tolist())
        list_predicted_scores['A'].extend(predicted_scores[:, 1].tolist())
        list_predicted_scores['D'].extend(predicted_scores[:, 2].tolist())

# Inserting the emotion scores in the dataset
for i in ['V', 'A', 'D']:
    df_data[i] = list_predicted_scores[i]

# Exporting the dataframe to a csv dataset
export_dataset = '../sentiment-model/test-posts-scores.csv'
df_data.to_csv(export_dataset, index=False)