import twint
import nest_asyncio
nest_asyncio.apply()

import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from transformers import BertTokenizer, BertForSequenceClassification
from nltk.corpus import stopwords
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
import re
import os


df = pd.read_csv('tweets.csv')  # tweets

# check GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('CPU:', torch.cuda.get_device_name(0))

PRE_TRAINED_MODEL_NAME = 'dbmdz/bert-base-turkish-128k-uncased'

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=True)

max_len = 100  # max kelime # pad arraydeki sayı max_len den fazlaysa kes, azsa 0 la doldur 

model = BertForSequenceClassification.from_pretrained('../bertmodel/')  # load model
model = model.to(device)  # for loaded model

df = df.rename(columns={'tweet': 'Comments'})

turkish_characters = "a|b|c|ç|d|e|f|g|ğ|h|ı|i|j|k|l|m|n|o|ö|p|r|s|ş|t|u|ü|v|y|z|0-9"
stop_words = set(stopwords.words("turkish"))

df["Comments"] = df["Comments"].astype(str)
df["Comments"] = df["Comments"].apply(lambda x: x.lower())
df["Comments"] = df["Comments"].apply((lambda x: re.sub('[^'+turkish_characters+'\s]','',x)))
df["Comments"] = df["Comments"].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
    

test_texts = df['Comments'].values


input_ids = []
attention_masks = []

for text in test_texts:
    encoded_dict = tokenizer.encode_plus(
                        text,                     
                        add_special_tokens = True, 
                        max_length = max_len,          
                        padding = 'max_length',
                        return_attention_mask = True,  
                        return_tensors = 'pt',   
                        truncation = True,
                   )
    
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0).to(device)
attention_masks = torch.cat(attention_masks, dim=0).to(device)

batch_size = 32  

prediction_data = TensorDataset(input_ids, attention_masks)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)


print('Prediction started on test data')
model.eval()
predictions = []


for batch in prediction_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask = batch

    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

    logits = outputs[0]
    logits = logits.detach().cpu().numpy()

    predictions.append(logits)
    
print('Prediction completed')

prediction_set = []

for i in range(len(predictions)):
    after_softmax = torch.nn.Softmax(dim=1)(torch.tensor(predictions[i]))
    for i in range(len(after_softmax)):
        pred_labels_i = after_softmax[i][1]
        prediction_set.append(pred_labels_i)
    
prediction_scores = np.array(prediction_set).tolist()


df['score'] = prediction_scores


score_plt = df.groupby('time').mean()['score']
date_plt = df.groupby('time').mean()['score'].index

print(f'Last Shape : {df.shape}')

dates = np.array(date_plt)
indices = np.argsort(dates)

dates = dates[indices][400:]
values = np.array(score_plt)[indices]
windows = pd.Series(values).rolling(400)
moving_averages = windows.mean()[400:]


plt.figure(figsize=(12,6))
plt.plot(dates, moving_averages, color='blue', label='Average Sentiment')
frequency = 500 
plt.xticks(dates[::frequency], dates[::frequency])

plt.axvline('', 0, 1, label='End of Derby Match', color='red', alpha=0.5)
plt.title(f'Analysis of Turkish Tweets about "{hangi_tweet_konusu}"')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.legend();
plt.show()