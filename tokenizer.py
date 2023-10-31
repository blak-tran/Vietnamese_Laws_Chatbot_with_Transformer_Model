import json, os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json


def save_token(tokenizer: Tokenizer, file_name: str = ""):
    path_save = "./checkpoints"
    
    if not os.path.exists(path_save):
        os.mkdir(path_save)
        
    tokenizer_json = tokenizer.to_json()
    with open(f'{path_save}/tokenizer_{file_name}.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))
    print("Tokenizer was saved!! ")
    
def tokenizer_QA_dataset(questions: list, answers: list, is_saved: bool = False) -> tf.keras.preprocessing.text.Tokenizer:
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(questions + answers)
    if is_saved:
        save_token(tokenizer)
    return tokenizer

def load_tokenizer_from_dict(tokenizer_dict):
    tokenizer = tokenizer_from_json(tokenizer_dict)
    return tokenizer

def load_tokenizer_from_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            tokenizer_dict = json.load(json_file)
        tokenizer = load_tokenizer_from_dict(tokenizer_dict)
        print("Tokenizer has been loaded! ")
        return tokenizer
    except FileNotFoundError:
        print(f"Tokenizer file not found at path: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from tokenizer file: {file_path}")
        return None