import numpy as np
import string
from collections import Counter

from keras.preprocessing.sequence import pad_sequences
from underthesea import word_tokenize
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import warnings
warnings.filterwarnings('ignore')

EMOTICONS = { 
        u":-3":"Happy face smiley",
        u":3":"Happy face smiley",
        u":->":"Happy face smiley",
        u":>":"Happy face smiley",
        u":))":"Happy face smiley",
        u":)))":"Happy face smiley",
        u":))))":"Happy face smiley",
        u":'<":"Happy face smiley",
        u":)":"Happy face smiley",
        u":(":"Happy face smiley",
        u":((":"Happy face smiley",
        u":‑D":"Laughing, big grin or laugh with glasses",
        u":D":"Laughing, big grin or laugh with glasses",
        u"XD":"Laughing, big grin or laugh with glasses",
        u"=D":"Laughing, big grin or laugh with glasses",
        u":‑c":"Frown, sad, andry or pouting",
        u":c":"Frown, sad, andry or pouting",
        u":‑<":"Frown, sad, andry or pouting",
        u":<":"Frown, sad, andry or pouting",
        u":@":"Frown, sad, andry or pouting",
        u"D:":"Sadness",
        u":O":"Surprise",
        u":o":"Surprise",
    }

def remove_emoticons(text):
    "Function to remove emoticons"
    arr = [word for word in text.split() if word not in EMOTICONS.keys()]
    return " ".join(arr)

def remove_rarewords(text, RAREWORDS):
    """custom function to remove the rare words"""
    return " ".join([word for word in str(text).split() if word not in RAREWORDS])

def preprocessing(df, RAREWORDS):
    df["question"] = df["question"].apply(lambda ele: ele.translate(str.maketrans('', '', string.punctuation))) # Remove punctuation
    df["answers"] = df["answers"].apply(lambda ele: ele.translate(str.maketrans('', '', string.punctuation))) 
    df["question"] = df["question"].apply(lambda ele: remove_emoticons(ele)) # Remove emoticons
    df["answers"] = df["answers"].apply(lambda ele: remove_emoticons(ele))
    df["question"] = df["question"].apply(lambda ele: remove_rarewords(ele, RAREWORDS)) # Remove rarewords
    df["answers"] = df["answers"].apply(lambda ele: remove_rarewords(ele, RAREWORDS))
    df['answers'] = df['answers'].apply(lambda ele: 'START ' + ele + ' END')
    df["question"] = df["question"].apply(lambda ele: ele.lower()) # convert text to lowercase
    df["answers"] = df["answers"].apply(lambda ele: ele.lower()) 
    
    return df

def encode_input_data(questions, tokenizer):
    #encoder_input_data
    questions = [str(q) for q in questions]
    tokenized_questions = [tokenizer.encode(question, add_special_tokens=True) for question in questions]
    maxlen_questions = max([len(x) for x in tokenized_questions])
    padded_questions = pad_sequences(tokenized_questions, maxlen = maxlen_questions, padding = 'post')
    encoder_input_data = np.array(padded_questions)
    return encoder_input_data, maxlen_questions

def decode_input_data(answers, tokenizer):
    # decoder_input_data
    answers = [str(q) for q in answers]
    tokenized_answers = [tokenizer.encode(answer, add_special_tokens=True) for answer in answers]
    maxlen_answers = max([len(x) for x in tokenized_answers])
    padded_answers = pad_sequences(tokenized_answers, maxlen = maxlen_answers, padding='post')
    decoder_input_data = np.array(padded_answers)
    return decoder_input_data, maxlen_answers

def data_processing(dataframe:str):
    
    idx = dataframe[dataframe['answers'].isnull()].index.tolist()  # Get index of nan row
    print('Question of nan answer: ', dataframe['question'][idx].values)
    
    # Fill in nan row value
    dataframe['answers'] = dataframe['answers'].fillna('Luật sư').values 

    cnt = Counter()
    for text in dataframe["answers"].values:
        for word in text.split():
            cnt[word] += 1
            
    RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-10-1:-1]])  # Get top 10 rare words
    
    dataframe = preprocessing(dataframe, RAREWORDS)
    
    # Convert DataFrame to NumPy array after preprocessing
    data = dataframe[['question', 'answers']].values
    
    questions = data[:, 0]  # convert question to a list
    answers = data[:, 1]    # convert answer that match with question to a list
    print("Questions: ", questions[:5]) 
    print("Answers: ",answers[:5])
    return questions, answers

class QADataset(TensorDataset):

    def __init__(self, questions, answers, max_len=128):
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.questions = questions
        self.answers = answers
        self.max_len = max_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]

        # Tokenize question and answer
        encoding = self.tokenizer.encode_plus(
            question,
            answer,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True
        )

        input_ids = encoding['input_ids']
        attn_masks = encoding['attention_mask']
        token_type_ids = encoding['token_type_ids']

        # Get start and end positions of the answer in the input sequence
        start_pos = input_ids.index(self.tokenizer.cls_token_id) + 1
        end_pos = input_ids.index(self.tokenizer.sep_token_id) - 1

        # Create labels indicating start and end positions of the answer
        start_label = [0] * self.max_len
        end_label = [0] * self.max_len
        start_label[start_pos] = 1
        end_label[end_pos] = 1

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attn_masks, dtype=torch.float32),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'start_positions': torch.tensor(start_label, dtype=torch.float32),
            'end_positions': torch.tensor(end_label, dtype=torch.float32)
        }
        