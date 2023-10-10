import numpy as np
import string
from gensim.models import KeyedVectors
from collections import Counter
from underthesea import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
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

def data_processing(dataframe, vector_path):
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
    
    fastText_model = KeyedVectors.load_word2vec_format(vector_path)
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
