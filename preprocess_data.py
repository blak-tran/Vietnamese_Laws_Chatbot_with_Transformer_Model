import pandas as pd
import numpy as np
import re,string
from gensim.models import KeyedVectors
from collections import Counter
from underthesea import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras import preprocessing, utils, activations
# from keras.callbacks import ModelCheckpoint
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
  df["user_a"] = df["user_a"].apply(lambda ele: ele.translate(str.maketrans('', '', string.punctuation))) # Remove punctuation
  df["user_b"] = df["user_b"].apply(lambda ele: ele.translate(str.maketrans('', '', string.punctuation))) 
  df["user_a"] = df["user_a"].apply(lambda ele: remove_emoticons(ele)) # Remove emoticons
  df["user_b"] = df["user_b"].apply(lambda ele: remove_emoticons(ele))
  df["user_a"] = df["user_a"].apply(lambda ele: remove_rarewords(ele, RAREWORDS)) # Remove rarewords
  df["user_b"] = df["user_b"].apply(lambda ele: remove_rarewords(ele, RAREWORDS))
  df['user_b'] = df['user_b'].apply(lambda ele: 'START ' + ele + ' END')
  df["user_a"] = df["user_a"].apply(lambda ele: ele.lower()) # convert text to lowercase
  df["user_b"] = df["user_b"].apply(lambda ele: ele.lower()) 
  
  return df

def encode_input_data(questions, tokenizer):
    #encoder_input_data
    tokenized_questions = tokenizer.texts_to_sequences(questions)
    maxlen_questions = max([len(x) for x in tokenized_questions])
    padded_questions = pad_sequences(tokenized_questions, maxlen = maxlen_questions, padding = 'post')
    encoder_input_data = np.array(padded_questions)
    print("Max length question:", maxlen_questions)
    print(encoder_input_data.shape)
    return encoder_input_data, maxlen_questions

def decode_input_data(answers, tokenizer):
    # decoder_input_data
    tokenized_answers = tokenizer.texts_to_sequences(answers)
    maxlen_answers = max([len(x) for x in tokenized_answers])
    padded_answers = pad_sequences(tokenized_answers, maxlen = maxlen_answers, padding='post')
    decoder_input_data = np.array(padded_answers)
    print("Max length anwser:", maxlen_answers)
    print(decoder_input_data.shape)
    return decoder_input_data, maxlen_answers

def data_processing(dataframe):
    idx = dataframe[dataframe['user_b'].isnull()].index.tolist() # Get index of nan row
    print('Question of nan answer: ' ,dataframe['user_a'][idx].values)
    # Fill in nan row value
    dataframe['user_b'] = dataframe['user_b'].fillna('Luật sư').values 

    cnt = Counter()
    for text in dataframe["user_b"].values:
        for word in text.split():
            cnt[word] += 1

    RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-10-1:-1]]) #Get top 10 rare word
    
    dataframe = preprocessing(dataframe, RAREWORDS)
    data = dataframe.values #numpy 
    questions = data[:,1] # convert question to a list
    answers = data[:,2] # convert answer that match with question to list
    print(questions[:5]) 
    print(answers[:5])
    
    # Tokenization questions
    questions = [word_tokenize(ques) for ques in questions]
    print(len(questions))
    print(questions[:3])
    
    # Tokenization answer
    answers = [word_tokenize(ans) for ans in answers]
    print(len(answers))
    print(answers[:3])
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(questions + answers)
    VOCAB_SIZE = len(tokenizer.word_index) + 2
    print( 'VOCAB SIZE : {}'.format( VOCAB_SIZE ))
    
    word2idx = tokenizer.word_index
    encoder_input_data, maxlen_questions = encode_input_data(questions=questions, tokenizer=tokenizer)
    decoder_input_data, maxlen_answers = decode_input_data(answers=answers, tokenizer=tokenizer)
    
    # decoder_output_data
    tokenized_answers = tokenizer.texts_to_sequences(answers)
    # Remove Start added before
    for i in range(len(tokenized_answers)):
        tokenized_answers[i] = tokenized_answers[i][1:]
    padded_answers = pad_sequences(tokenized_answers, maxlen = maxlen_answers, padding='post')
    decoder_output_data = np.array(padded_answers)
    print(decoder_output_data.shape)
    
    fastText_model = KeyedVectors.load_word2vec_format('input/wiki-vi-vectors/wiki.vi.vec')
    print("FastText Loaded!")
    
    embeddings_dim = 300

    embedding_matrix = np.zeros((VOCAB_SIZE, embeddings_dim))

    for word, index in word2idx.items():
        try:
            embedding_matrix[index,:] = fastText_model[word]
        except:
            continue
            
    print(embedding_matrix.shape)
    
    return encoder_input_data,decoder_input_data, decoder_output_data,maxlen_answers,embedding_matrix,maxlen_questions, VOCAB_SIZE, embeddings_dim
