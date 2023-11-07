import numpy as np
import tensorflow as tf
from underthesea import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models import transformer
from gensim.models import KeyedVectors
import pandas as pd 
from collections import Counter
from preprocess_data import preprocessing
from tokenizer import load_tokenizer_from_json


embeddings_dim = 300

maxlen_answers = 44
maxlen_questions = 76

# Hyper-parameters
NUM_LAYERS = 2
NUM_HEADS = 6 # model dims must divided to number of heads
UNITS = 512
DROPOUT = 0.1


dataframe = pd.read_csv("/home/dattran/datadrive2/AI-project/vietnamese_chatbot_research/input/vietnamese-chatbot/mini_batches/qa_batch_0.csv")
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

idx2word = tokenizer.index_word
word2idx = tokenizer.word_index
VOCAB_SIZE = len(tokenizer.word_index) + 2  

embedding_matrix = np.zeros((VOCAB_SIZE, embeddings_dim))

fastText_model = KeyedVectors.load_word2vec_format("/home/dattran/datadrive2/AI-project/vietnamese_chatbot_research/input/wiki-vi-vectors/wiki.vi.vec")
print("FastText Loaded!")
for word, index in word2idx.items():
    try:
        embedding_matrix[index,:] = fastText_model[word]
    except:
        continue
        
print(embedding_matrix.shape)
    
    
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


model.load_weights('/home/dattran/datadrive2/AI-project/vietnamese_chatbot_research/old_checkpoint_chatbot/2023_10_11_11_58_07_qa_batch_0/2023_10_11_11_58_07_qa_batch_0_3600.h5')

# Check the loaded model architecture
model.summary()

def str_to_tokens(sentence):
    words = word_tokenize(sentence.lower())
    tokens_list = []
    
    for word in words:
        tokens_list.append(word2idx[word]) 
    return pad_sequences([tokens_list],maxlen = maxlen_questions , padding='post')


def evaluate(sentence):
  sentence = str_to_tokens(sentence)
  output = np.zeros((1, 1))
  output[0, 0] = word2idx['start']

  for i in range(maxlen_answers):
    predictions = model(inputs=[sentence, output], training=False)

    # select the last word from the seq_len dimension
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # return the result if the predicted_id is equal to the end token
    if tf.equal(predicted_id, word2idx['end']):
      break

    # concatenated the predicted_id to the output which is given to the decoder as its input.
    output = tf.concat([output, predicted_id], axis=-1)
    
  return tf.squeeze(output, axis=0)

def predict(sentence):
  prediction = evaluate(sentence)
  predicted_sentence = " ".join(idx2word[tf.get_static_value(i)] for i in prediction if i < VOCAB_SIZE)
  return predicted_sentence.replace("start","")


flag=True
print("BOT: Xin chào! Tôi là ChatBot. Nếu bạn muốn ngưng cuộc trò chuyện, hãy gõ Bye!")

while(flag==True):
    human_response = input('Enter question : ')
    if human_response != 'bye':
        try:
            print('BOT: ' + predict(human_response))
        except:
            print("BOT: Xin lỗi câu này tôi chưa đc học ,vui lòng hỏi lại :( ")
    else:
        flag=False
        print("BOT: Tạm biệt nha!")
