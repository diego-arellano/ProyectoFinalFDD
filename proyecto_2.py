#%%
import pandas as pd
import json 
import string
from nltk.corpus import stopwords
import numpy as np
#%%
def jaccard_distnace():
    assert NotImplementedError

#%%
stop = stopwords.words('english')

with open('train-v2.0.json','r') as f:
    data = json.loads(f.read())

train = pd.json_normalize(data, record_path = ['data', 'paragraphs', 'qas', 'answers'], 
meta = [['data', 'title'], ['data', 'paragraphs', 'context'], ['data', 'paragraphs', 'qas', 'question'], ['data', 'paragraphs', 'qas', 'is_impossible']])

train.rename(columns={'text':'answer', 'data.paragraphs.context':'context', 'data.title':'title', 'data.paragraphs.qas.question':'question', 'data.paragraphs.qas.is_impossible':'is_impossible'}, inplace=True)

train['amswer'] = train['answer'].str.lower()
train['question'] = train['question'].str.lower()
train['context'] = train['context'].str.lower()

train['context'] = train['context'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
train['context'] = train['context'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

'''
train["context"]= train["context"].str.split(".", expand = True)
jaccard_dist_df = pd.Dataframe({"question":pd.Series(dtype="str"),"sentence":pd.Series(dtype="str"),"jaccard_distance":pd.Series(dtype=np.float64)})
for row in train:
    sentences = row["context"].split(".")
    for sentence in sentences:
        new_row = {'question':row['question'], 'sentence':sentence, 'jaccard_distance':92}
        jaccard_dist_df = jaccard_dist_df.append(new_row, ignore_index=True)

print(train.head(10))
'''
# %%
