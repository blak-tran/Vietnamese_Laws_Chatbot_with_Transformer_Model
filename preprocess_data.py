import numpy as np
import string, math, os
from gensim.models import KeyedVectors
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from underthesea import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tokenizer import tokenizer_QA_dataset, load_tokenizer_from_json, save_token
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
    df["answer"] = df["answer"].apply(lambda ele: ele.translate(str.maketrans('', '', string.punctuation))) 
    df["question"] = df["question"].apply(lambda ele: remove_emoticons(ele)) # Remove emoticons
    df["answer"] = df["answer"].apply(lambda ele: remove_emoticons(ele))
    df["question"] = df["question"].apply(lambda ele: remove_rarewords(ele, RAREWORDS)) # Remove rarewords
    df["answer"] = df["answer"].apply(lambda ele: remove_rarewords(ele, RAREWORDS))
    df['answer'] = df['answer'].apply(lambda ele: 'START ' + ele + ' END')
    df["question"] = df["question"].apply(lambda ele: ele.lower()) # convert text to lowercase
    df["answer"] = df["answer"].apply(lambda ele: ele.lower()) 
    
    return df

def encode_input_data(questions, tokenizer):
    #encoder_input_data
    tokenized_questions = tokenizer.texts_to_sequences(questions)
    maxlen_questions = max([len(x) for x in tokenized_questions])
    padded_questions = pad_sequences(tokenized_questions, maxlen = maxlen_questions, padding = 'post')
    encoder_input_data = np.array(padded_questions)
    return encoder_input_data, maxlen_questions

def decode_input_data(answers, tokenizer):
    # decoder_input_data
    tokenized_answers = tokenizer.texts_to_sequences(answers)
    # maxlen_answers = max([len(x) for x in tokenized_answers])
    maxlen_answers = 512
    padded_answer = pad_sequences(tokenized_answers, maxlen = maxlen_answers, padding='post')
    decoder_input_data = np.array(padded_answer)
    return decoder_input_data, maxlen_answers

def data_processing(dataframe:str , 
                    vector_path: str = None,
                    config: str = None,
                    checkpoint_name: str = None):
    
    idx = dataframe[dataframe['answer'].isnull()].index.tolist()  # Get index of nan row
    print('Question of nan answer: ', dataframe['question'][idx].values)
    
    # Fill in nan row value
    dataframe['answer'] = dataframe['answer'].fillna('Luật sư').values 

    cnt = Counter()
    for text in dataframe["answer"].values:
        for word in text.split():
            cnt[word] += 1
            
    RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-10-1:-1]])  # Get top 10 rare words
    
    dataframe = preprocessing(dataframe, RAREWORDS)
    dataframe = dataframe[:1000]
        
    # Convert DataFrame to NumPy array after preprocessing
    data = dataframe.values #numpy 
    questions = data[:,0] # convert question to a list
    answers = data[:,1] # convert answer that match with question to list
    print(questions[:5]) 
    print(answers[:5])
    
    questions = [word_tokenize(ques) for ques in questions]
    print("Len of questions: ", len(questions))
    print("words tokenize questions: ", questions[:3])

    # Tokenization answer
    answers = [word_tokenize(ans) for ans in answers]
    print("Len of answers: ", len(answers))
    
    print("words tokenize answer: ",answers[:3])
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(questions + answers)
    VOCAB_SIZE = len(tokenizer.word_index) + 2
    print( 'VOCAB SIZE : {}'.format( VOCAB_SIZE ))
    
    
    word2idx = tokenizer.word_index
    encoder_input_data, maxlen_questions = encode_input_data(questions=questions, tokenizer=tokenizer)
    decoder_input_data, maxlen_answers = decode_input_data(answers=answers, tokenizer=tokenizer)
    
    
    fastText_model = KeyedVectors.load_word2vec_format(vector_path)
    print("FastText Loaded!")

    embeddings_dim = 300
    
    # decoder_output_data
    tokenized_answers = tokenizer.texts_to_sequences(answers)
    # Remove Start added before
    for i in range(len(tokenized_answers)):
        tokenized_answers[i] = tokenized_answers[i][1:]
    padded_answers = pad_sequences(tokenized_answers, maxlen = maxlen_answers, padding='post')
    decoder_output_data = np.array(padded_answers)
    print(decoder_output_data.shape)

    embedding_matrix = np.zeros((VOCAB_SIZE, embeddings_dim))

    for word, index in word2idx.items():
        try:
            embedding_matrix[index,:] = fastText_model[word]
        except:
            continue
            
    print(embedding_matrix.shape)
    
    return encoder_input_data,decoder_input_data, decoder_output_data, maxlen_answers,embedding_matrix,maxlen_questions, VOCAB_SIZE, embeddings_dim



def create_batches(data, batch_size=10000):
    num_batches = math.ceil(len(data) / batch_size)
    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batches.append(data[start_idx:end_idx])
    return batches

def process_data_for_training(dataframe, output_dir):
    idx = dataframe[dataframe['answer'].isnull()].index.tolist()  # Get index of nan row
    print('Question of nan answer: ', dataframe['question'][idx].values)
    
    # Fill in nan row value
    dataframe['answer'] = dataframe['answer'].fillna('Luật sư').values 

    cnt = Counter()
    for text in dataframe["answer"].values:
        for word in text.split():
            cnt[word] += 1

    data = dataframe[['question', 'answer']].values
    
    questions = data[:, 0]  # convert question to a list
    answer = data[:, 1]    # convert answer that match with question to a list
    print(questions[:5]) 
    print(answer[:5])

    qa_pairs = list(zip(questions, answer))
    batches = create_batches(qa_pairs, batch_size=6000)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, batch in enumerate(batches):
        output_filename = os.path.join(output_dir, f'qa_batch_{idx}.csv')
        questions_batch, answer_batch = zip(*batch)
        df_batch = pd.DataFrame({'question': questions_batch, 'answer': answer_batch})
        df_batch.to_csv(output_filename, index=False)

    return output_dir

if __name__ == "__main__":
    dataframe = pd.read_csv("/home/dattran/datadrive/research/data/chatbot/d liu chatbot question-answer short style.csv")
    # out_dir = "/datadrive2/AI-project/vietnamese_chatbot_research/input/vietnamese-chatbot/mini_batches"
    # output_directory = process_data_for_training(dataframe, out_dir)
    # print(f"Mini-batches created and saved in: {output_directory}")
    tokenizer = data_processing(dataframe=dataframe, vector_path="/home/dattran/datadrive/research/data/chatbot/wiki.vi.vec")
    print(tokenizer)
    