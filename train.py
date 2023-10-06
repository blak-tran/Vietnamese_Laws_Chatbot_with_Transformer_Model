import pandas as pd
import numpy as np
import os 
from gensim.models import KeyedVectors
from collections import Counter
from underthesea import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from preprocess_data import preprocessing
from models import transformer
from loss import CustomSchedule, loss_function, accuracy
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
# from keras.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)

AUTOTUNE = tf.data.experimental.AUTOTUNE
    
print(tf.__version__)

# Hyper-parameters
NUM_LAYERS = 2
NUM_HEADS = 6 # model dims must divided to number of heads
UNITS = 512
DROPOUT = 0.1

def train(encoder_input_data, decoder_input_data, decoder_output_data, maxlen_answers, embedding_matrix, maxlen_questions, VOCAB_SIZE, embeddings_dim):
    tf.keras.backend.clear_session()
    epochs = 10000
    checkpoint_dir = "./checkpoints"
    
    # Create the checkpoint directory if it does not exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define the model
    model = transformer(
        maxlen_answers=maxlen_answers,
        embedding_matrix=embedding_matrix,
        maxlen_questions=maxlen_questions,
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=embeddings_dim,
        num_heads=NUM_HEADS,
        dropout=DROPOUT)
    
    learning_rate = CustomSchedule(embeddings_dim)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

    # Set up TensorBoard logging
    tensorboard_callback = TensorBoard(log_dir="./logs", histogram_freq=1)

    # Set up ModelCheckpoint callback to save model weights every 1000 steps
    checkpoint_callback = ModelCheckpoint(
        os.path.join(checkpoint_dir, "checkpoint_{epoch}.h5"),
        save_freq=1000,  # Save every 1000 steps
        save_weights_only=True,
        verbose=1
    )

    # Training the model with TensorBoard and ModelCheckpoint callbacks
    history = model.fit([encoder_input_data, decoder_input_data], decoder_output_data, batch_size=128, epochs=epochs, callbacks=[tensorboard_callback, checkpoint_callback])

    return history


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

dataframe = pd.read_csv("input/vietnamese-chatbot/d liu chatbot question-answer short style.csv")
encoder_input_data,decoder_input_data, decoder_output_data,maxlen_answers,embedding_matrix,maxlen_questions, VOCAB_SIZE, embeddings_dim = data_processing(dataframe=dataframe)
train(encoder_input_data,decoder_input_data, decoder_output_data,maxlen_answers,embedding_matrix,maxlen_questions, VOCAB_SIZE, embeddings_dim)

