#%%
import pandas as pd
import json 
import nltk
import numpy as np
nltk.download('stopwords')
nltk.download('punkt')

def jaccard_distance(a, b):
    assert NotImplementedError
    print(a,b)
    intersection = len(list(set(a).intersection(b)))
    union = (len(a) + len(b)) - intersection
    if union == 0: 
        return np.inf
    return float(intersection) / union



stop = nltk.corpus.stopwords.words('english')

#%%

with open('train-v2.0.json','r') as f:
    data = json.loads(f.read())


train = pd.json_normalize(data, record_path = ['data', 'paragraphs', 'qas', 'answers'], 
meta = [['data', 'title'], ['data', 'paragraphs', 'context'], ['data', 'paragraphs', 'qas', 'question'], ['data', 'paragraphs', 'qas', 'is_impossible']])

train.rename(columns={'text':'answer', 'data.paragraphs.context':'context', 'data.title':'title', 'data.paragraphs.qas.question':'question', 'data.paragraphs.qas.is_impossible':'is_impossible'}, inplace=True)

train['amswer'] = train['answer'].str.lower()
train['question'] = train['question'].str.lower()
train['context'] = train['context'].str.lower()

train['context'] = train['context'].str.replace(r'[?!,\'()]+', '')

train['context'] = train['context'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
train['question'] = train['question'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

train['context'] = train['context'].str.split(".")

train = train.explode('context') 
train['context'] = train['context'].apply(lambda x: nltk.tokenize.word_tokenize(x))
train['question'] = train['question'].apply(lambda x: nltk.tokenize.word_tokenize(x))

train = train[train['context'].map(lambda x: len(x)) > 0]

#%%

jaccard_df = pd.DataFrame()

train_short = train.loc[0]

jaccard_df["Jaccard(context_i_line_j, q_i_k)"] = train_short.apply(lambda x: jaccard_distance(x['context'], x['question']), axis=1)

print(jaccard_df["Jaccard(context_i_line_j, q_i_k)"].head(10))

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
