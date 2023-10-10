from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import pandas as pd
from underthesea import word_tokenize
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer 
import warnings
warnings.filterwarnings('ignore')

from preprocess_data import preprocessing




def word_to_vec(questions, answers, RAREWORDS, vector_size=300, window=5, min_count=1, workers=4):
    # Combine questions and answers into a single list for Word2Vec training
    combined_sentences = questions + answers
    
    # Preprocess the combined sentences
    # processed_sentences = [preprocessing(sentence, RAREWORDS) for sentence in combined_sentences]
    
    # Train Word2Vec model
    word2vec_model = Word2Vec(sentences=combined_sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    
    return word2vec_model


def data_processing(dataframe):
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
    
    return questions, answers, RAREWORDS


if __name__ == "__main__":
    data_file = "/home/dattran/datadrive/AI-project/vietnamese_chatbot_research/input/vietnamese-chatbot/vi-QA.csv"
    dataframe = pd.read_csv(data_file)
    question_tokens, answer_tokens, RAREWORDS = data_processing(dataframe)
    
    word2vec_model = word_to_vec(question_tokens, answer_tokens, RAREWORDS)
    
    word_vectors = word2vec_model.wv
    # Save word vectors to a vi.vec file
    # Save word vectors to a vi.vec file
    with open("/home/dattran/datadrive/AI-project/vietnamese_chatbot_research/input/wiki-vi-vectors/QA.vi.vec", "w", encoding="utf-8") as f:
        # Write the header with the vocabulary size and vector size
        vocab_size, vector_size = len(word_vectors.index_to_key), word_vectors.vector_size
        f.write(f"{vocab_size} {vector_size}\n")
        
        # Write word vectors to the file
        for word in word_vectors.index_to_key:
            vector = word_vectors[word]
            vector_str = " ".join(str(value) for value in vector)
            f.write(f"{word} {vector_str}\n")


    