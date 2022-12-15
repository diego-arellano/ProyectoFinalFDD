#%%
import pandas as pd
import json 
import nltk
import numpy as np
import regex as re
nltk.download('stopwords')
nltk.download('punkt')
import logit as lg


def jaccard_distance(a, b):
    assert NotImplementedError
    #print(a,b)
    intersection = len(list(set(a).intersection(b)))
    union = (len(a) + len(b)) - intersection
    if union == 0: 
        return np.inf
    return float(intersection) / union

def remove_accents(s):
    replacements = [
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
    ]
    for a, b in replacements:
        s = s.replace(a, b)
    return s

def contains_answer(context_sentence, answer, impossible):
    if impossible: 
        return 0
    return int(set(answer).issubset(set(context_sentence)))


stop = nltk.corpus.stopwords.words('english')

#%%

with open('train-v2.0.json','r') as f:
    data = json.loads(f.read())


train = pd.json_normalize(data, record_path = ['data', 'paragraphs', 'qas', 'answers'], 
meta = [['data', 'title'], ['data', 'paragraphs', 'context'], ['data', 'paragraphs', 'qas', 'question'], ['data', 'paragraphs', 'qas', 'is_impossible']])

train.rename(columns={'text':'answer', 'data.paragraphs.context':'context', 'data.title':'title', 'data.paragraphs.qas.question':'question', 'data.paragraphs.qas.is_impossible':'is_impossible'}, inplace=True)

train['answer'] = train['answer'].str.lower()
train['question'] = train['question'].str.lower()
train['context'] = train['context'].str.lower()

train['answer'] = train['answer'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
train['context'] = train['context'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
train['question'] = train['question'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

train['answer'] = train['answer'].apply(lambda x: remove_accents(x))
train['context'] = train['context'].apply(lambda x: remove_accents(x))
train['question'] = train['question'].apply(lambda x: remove_accents(x))

train['answer'] = train['answer'].apply(lambda x: re.sub("[^\w .]", "", x))
train['context'] = train['context'].apply(lambda x: re.sub("[^\w .]", "", x))
train['question'] = train['question'].apply(lambda x: re.sub("[^\w .]", "", x))

train['context'] = train['context'].str.split(".")

train = train.explode('context')
train['answer'] = train['answer'].apply(lambda x: nltk.tokenize.word_tokenize(x))
train['context'] = train['context'].apply(lambda x: nltk.tokenize.word_tokenize(x))
train['question'] = train['question'].apply(lambda x: nltk.tokenize.word_tokenize(x))


train = train[train['context'].map(lambda x: len(x)) > 0]

#%%
jaccard_df = pd.DataFrame()

jaccard_df["Jaccard(context_i_line_j, q_i_k)"] = train.apply(lambda x: jaccard_distance(x['context'], x['question']), axis=1)

jaccard_df["Ans"] = train.apply(lambda x: contains_answer(x['context'], x['answer'], x['is_impossible']), axis=1)

jaccard_df = jaccard_df.reset_index()
#%%


sample_size_5 = int(len(jaccard_df)*0.05)
sample_size_1 = int(len(jaccard_df)*0.01)

s_data_5 = jaccard_df.sample(sample_size_5, random_state=123454321)
s_data_1 = jaccard_df.sample(sample_size_1, random_state=123454321)

sample_X_5 = s_data_5.iloc[:,:].values
print(sample_X_5)
#sample_y_5 = 

#sample_X_1 = 
#sample_y_1 = 
# %%
