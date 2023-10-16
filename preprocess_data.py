import numpy as np
import string, math, os
from gensim.models import KeyedVectors
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from underthesea import word_tokenize
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
    df["answers"] = df["answers"].apply(lambda ele: ele.translate(str.maketrans('', '', string.punctuation))) 
    df["question"] = df["question"].apply(lambda ele: remove_emoticons(ele)) # Remove emoticons
    df["answers"] = df["answers"].apply(lambda ele: remove_emoticons(ele))
    df["question"] = df["question"].apply(lambda ele: remove_rarewords(ele, RAREWORDS)) # Remove rarewords
    df["answers"] = df["answers"].apply(lambda ele: remove_rarewords(ele, RAREWORDS))
    df['answers'] = df['answers'].apply(lambda ele: 'START ' + ele + ' END')
    df["question"] = df["question"].apply(lambda ele: ele.lower()) # convert text to lowercase
    df["answers"] = df["answers"].apply(lambda ele: ele.lower()) 
    
    return df

def encode_input_data(questions, tokenizer, maxlen_questions):
    #encoder_input_data
    tokenized_questions = tokenizer.texts_to_sequences(questions)
    padded_questions = pad_sequences(tokenized_questions, maxlen = maxlen_questions, padding = 'post')
    encoder_input_data = np.array(padded_questions)
    return encoder_input_data, maxlen_questions

def decode_input_data(answers, tokenizer, maxlen_answers):
    # decoder_input_data
    tokenized_answers = tokenizer.texts_to_sequences(answers)
    padded_answers = pad_sequences(tokenized_answers, maxlen = maxlen_answers, padding='post')
    decoder_input_data = np.array(padded_answers)
    return decoder_input_data, maxlen_answers

def data_processing(dataframe:str , 
                    vector_path: str = None,
                    config: str = None,
                    checkpoint_name: str = None):
    
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
    
    questions = [word_tokenize(ques) for ques in questions]
    print(len(questions))
    print("words tokenize questions: ", questions[:3])

    # Tokenization answer
    answers = [word_tokenize(ans) for ans in answers]
    print(len(answers))
    print("words tokenize answers: ",answers[:3])
    
    VOCAB_SIZE = 4512
    maxlen_questions = 76
    maxlen_answers = 44
    embeddings_dim = 300
    
    print( 'VOCAB SIZE : {}'.format( VOCAB_SIZE ))
    
    if config == None:
        tokenizer = tokenizer_QA_dataset(questions=questions, answers=answers, is_saved=True)
        encoder_input_data, _ = encode_input_data(questions=questions, tokenizer=tokenizer, maxlen_questions=maxlen_questions)
        decoder_input_data, _ = decode_input_data(answers=answers, tokenizer=tokenizer, maxlen_answers=maxlen_answers)

    else:
        tokenizer = load_tokenizer_from_json(config)
        # Tokenize the new questions and answers
        
        tokenized_new_questions = tokenizer.texts_to_sequences(questions)
        tokenized_new_answers = tokenizer.texts_to_sequences(answers)
        # Flatten the list of tokenized words
        flattened_new_questions = [word for sublist in tokenized_new_questions for word in sublist]
        flattened_new_answers = [word for sublist in tokenized_new_answers for word in sublist]

        # Update the tokenizer's word counts
        tokenizer.word_counts.update(Counter(flattened_new_questions + flattened_new_answers))

        # Update the tokenizer's document count
        tokenizer.document_count += len(flattened_new_questions + flattened_new_answers)

        # Update the tokenizer's index_docs dictionary
        tokenizer.index_docs.update({index + len(tokenizer.word_index): 1 for index in range(len(tokenizer.word_index), len(tokenizer.word_index) + len(questions + answers))})
        
        # Pad the tokenized sequences
        encoder_input_data = pad_sequences(tokenized_new_questions, maxlen=maxlen_questions, padding='post')
        decoder_input_data = pad_sequences(tokenized_new_answers, maxlen=maxlen_answers, padding='post')

        save_token(tokenizer, checkpoint_name)
    
    word2idx = tokenizer.word_index
    
    # Check for NaN or Inf values in encoder_input_data
    nan_encoder_input_data = np.isnan(encoder_input_data)
    inf_encoder_input_data = np.isinf(encoder_input_data)

    # Check for NaN or Inf values in decoder_input_data
    nan_decoder_input_data = np.isnan(decoder_input_data)
    inf_decoder_input_data = np.isinf(decoder_input_data)

    # Print the indices of NaN or Inf values
    print("NaN values in encoder_input_data:", np.argwhere(nan_encoder_input_data))
    print("Inf values in encoder_input_data:", np.argwhere(inf_encoder_input_data))

    print("NaN values in decoder_input_data:", np.argwhere(nan_decoder_input_data))
    print("Inf values in decoder_input_data:", np.argwhere(inf_decoder_input_data))
    
    print("Max length question:", maxlen_questions)
    print("Encoder input data shape: ",encoder_input_data.shape)
    
    print("Max length anwser:", maxlen_answers)
    print("Decoder input data shape: ",decoder_input_data.shape)
    
    # decoder_output_data
    tokenized_answers = tokenizer.texts_to_sequences(answers)
    # Remove Start added before
    for i in range(len(tokenized_answers)):
        tokenized_answers[i] = tokenized_answers[i][1:]
    padded_answers = pad_sequences(tokenized_answers, maxlen = maxlen_answers, padding='post')
    decoder_output_data = np.array(padded_answers)
    
    nan_decoder_output_data = np.isnan(decoder_output_data)
    inf_decoder_output_data = np.isinf(decoder_output_data)

    # Print the indices of NaN or Inf values
    print("NaN values in decoder_output_data:", np.argwhere(nan_decoder_output_data))
    print("Inf values in decoder_output_data:", np.argwhere(inf_decoder_output_data))
    print(decoder_output_data.shape)
    
    fastText_model = KeyedVectors.load_word2vec_format(vector_path)
    print("FastText Loaded!")

    embedding_matrix = np.zeros((VOCAB_SIZE, embeddings_dim))

    for word, index in word2idx.items():
        try:
            embedding_matrix[index,:] = fastText_model[word]
        except:
            continue# Assuming embedding_matrix is your embedding matrix
    has_nan = np.isnan(embedding_matrix).any()
    has_inf = np.isinf(embedding_matrix).any()

    if has_nan:
        print("Embedding matrix contains NaN values.")
        embedding_matrix[np.isnan(embedding_matrix)] = 0  # Replace NaN with 0
    else:
        print("Embedding matrix does not contain NaN values.")

    if has_inf:
        print("Embedding matrix contains Inf values.")
        embedding_matrix[np.isinf(embedding_matrix)] = 0
    else:
        print("Embedding matrix does not contain Inf values.")
    
    print(embedding_matrix.shape)
    
    return encoder_input_data,decoder_input_data, decoder_output_data,maxlen_answers,embedding_matrix,maxlen_questions, VOCAB_SIZE, embeddings_dim



def create_batches(data, batch_size=10000):
    num_batches = math.ceil(len(data) / batch_size)
    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batches.append(data[start_idx:end_idx])
    return batches

def process_data_for_training(dataframe, output_dir):
    idx = dataframe[dataframe['answers'].isnull()].index.tolist()  # Get index of nan row
    print('Question of nan answer: ', dataframe['question'][idx].values)
    
    # Fill in nan row value
    dataframe['answers'] = dataframe['answers'].fillna('Luật sư').values 

    cnt = Counter()
    for text in dataframe["answers"].values:
        for word in text.split():
            cnt[word] += 1

    data = dataframe[['question', 'answers']].values
    
    questions = data[:, 0]  # convert question to a list
    answers = data[:, 1]    # convert answer that match with question to a list
    print(questions[:5]) 
    print(answers[:5])

    qa_pairs = list(zip(questions, answers))
    batches = create_batches(qa_pairs, batch_size=6000)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, batch in enumerate(batches):
        output_filename = os.path.join(output_dir, f'qa_batch_{idx}.csv')
        questions_batch, answers_batch = zip(*batch)
        df_batch = pd.DataFrame({'question': questions_batch, 'answers': answers_batch})
        df_batch.to_csv(output_filename, index=False)

    return output_dir

if __name__ == "__main__":
    dataframe = pd.read_csv("/datadrive2/AI-project/vietnamese_chatbot_research/input/vietnamese-chatbot/vi-QA.csv")
    # out_dir = "/datadrive2/AI-project/vietnamese_chatbot_research/input/vietnamese-chatbot/mini_batches"
    # output_directory = process_data_for_training(dataframe, out_dir)
    # print(f"Mini-batches created and saved in: {output_directory}")
    tokenizer = data_processing(dataframe=dataframe, save_tokenizer=True)
    print(tokenizer)
    