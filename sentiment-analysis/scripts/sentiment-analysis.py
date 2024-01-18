import os
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification
from tqdm import tqdm

# Importing the csv dataset.
import_dataset = 'datasets/ishw-case/mastodon/mastodon-israel-cleaned.csv'
df_data = pd.read_csv(import_dataset)
print(len(df_data))

# Preparing the dataset for the network this includes tokenization, encoding and creating dataloaders.
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

encodings = tokenizer(df_data['text'].tolist(), truncation=True, padding=True, return_tensors='pt')

dataset = torch.utils.data.TensorDataset(
    encodings['input_ids'], 
    encodings['attention_mask']
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Loading the model.
model= torch.load('sentiment-analysis/models/bert-imdb.pth')

# Using the model to perform sentiment analysis on the dataset.
model.eval()
list_predicted_scores = []

for batch in tqdm(dataloader):
    with torch.no_grad():
        input_ids, attention_mask = batch

        # Obtaining the sentiment score.
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        predicted_scores = output.logits.view(-1)

        # Writing the sentiment score to the list.
        list_predicted_scores.extend(predicted_scores.tolist())

# Inserting the sentiment score in the dataset.
df_data['s'] = list_predicted_scores

# Exporting the dataframe to a csv dataset.
export_dataset = 'datasets/ishw-case/scored/isreal-scored.csv'
df_data.to_csv(export_dataset, index=False)