import argparse
import os, pytz
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
from datetime import datetime

import pandas as pd
import tensorflow as tf
from preprocess_data import data_processing
from models import transformer
from loss import CustomSchedule, loss_function, accuracy
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback


def parse_args():
    parser = argparse.ArgumentParser(description='Training script for chatbot model.')
    parser.add_argument('--epochs', type=int, default=1200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--embedding_dim', type=int, default=300, help='Dimensionality of word embeddings')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save training logs')
    parser.add_argument('--data_file', type=str, default='./input/vietnamese-chatbot/vi-QA.csv', help='Path to CSV file containing training data')
    parser.add_argument('--save_frequency', type=int, default=600, help="Model checkpoint will be saved every epochs th")
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the pre-trained model checkpoint')
    parser.add_argument('--vector_path', type=str, default="", help='Path to vector file')
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--units', type=int, default=512)
    parser.add_argument('--config', type=str, default="./checkpoints/tokenizer.json")
    
    
    return parser.parse_args()

def setup_tpu():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Device:', tpu.master())
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    except:
        strategy = tf.distribute.get_strategy()
    print('Number of replicas:', strategy.num_replicas_in_sync)
    return strategy

class CustomModelCheckpoint(Callback):
    def __init__(self, filepath, save_frequency):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.save_frequency = save_frequency

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_frequency == 0:
            self.model.save_weights(self.filepath.format(epoch=epoch + 1), overwrite=True)
            print(f"Saved model weights for epoch {epoch + 1}.")

def load_model_weights(checkpoint_path, model):
    model.load_weights(checkpoint_path)
    print(f"Loaded checkpoint weights from {checkpoint_path}")
    return model
     
def train_model(checkpoint_name, args, encoder_input_data, decoder_input_data, decoder_output_data, maxlen_answers, embedding_matrix, maxlen_questions, VOCAB_SIZE, embeddings_dim):
    tf.keras.backend.clear_session()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    

    # Define the model
    model = transformer(
        maxlen_answers=maxlen_answers,
        embedding_matrix=embedding_matrix,
        maxlen_questions=maxlen_questions,
        vocab_size=VOCAB_SIZE,
        num_layers=args.num_layers,
        units=args.units,
        d_model=embeddings_dim,
        num_heads=args.num_heads,
        dropout=args.dropout)
    
    model.summary()

    learning_rate = CustomSchedule(embeddings_dim)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

    # Set up TensorBoard logging
    print("Training log dir: ", args.log_dir)
    tensorboard_callback = TensorBoard(log_dir=args.log_dir, histogram_freq=1)

    # Set up ModelCheckpoint callback to save model weights
    root_batch_path = f"./checkpoints/{checkpoint_name}"
    if not os.path.exists(root_batch_path):
        os.makedirs(root_batch_path)
        
    checkpoint_filepath = f"{root_batch_path}/{checkpoint_name}_{{epoch}}.h5"
    print("Save dir: ", checkpoint_filepath)
    print("Save_frequency: ",int(args.save_frequency))
        
    custom_checkpoint_callback = CustomModelCheckpoint(filepath=checkpoint_filepath, save_frequency=int(args.save_frequency))

    
    if args.checkpoint_path is not None:
        # Load the checkpoint weights
        model = load_model_weights(args.checkpoint_path, model)
        print(f"Loaded checkpoint weights from {args.checkpoint_path}")
        
        
    # Training the model with TensorBoard and ModelCheckpoint callbacks
    history = model.fit([encoder_input_data, decoder_input_data], decoder_output_data, batch_size=args.batch_size, epochs=args.epochs, callbacks=[tensorboard_callback, custom_checkpoint_callback])

    return history

def get_logger_datetime():
    # Get the current datetime in UTC
    current_datetime = datetime.now(pytz.utc)

    # Convert to Vietnamese time zone (Asia/Ho_Chi_Minh)
    vietnamese_tz = pytz.timezone('Asia/Ho_Chi_Minh')
    current_datetime_vietnamese = current_datetime.astimezone(vietnamese_tz)
    
    logger_datetime = current_datetime_vietnamese.strftime("%Y_%m_%d_%H_%M_%S")

    return logger_datetime

def main():
    args = parse_args()
    
    current_datetime = get_logger_datetime() 

    base_name = os.path.splitext(os.path.basename(args.data_file))[0]

    print("Base Name:", base_name)
    checkpoint_name = f"{current_datetime}_{base_name}"
    dataframe = pd.read_csv(args.data_file)
    
    encoder_input_data, decoder_input_data, decoder_output_data, maxlen_answers, embedding_matrix, maxlen_questions, VOCAB_SIZE, embeddings_dim = data_processing(dataframe=dataframe, vector_path=args.vector_path, config=args.config, checkpoint_name=checkpoint_name)
    
    train_model(checkpoint_name, args, encoder_input_data, decoder_input_data, decoder_output_data, maxlen_answers, embedding_matrix, maxlen_questions, VOCAB_SIZE, embeddings_dim)

if __name__ == "__main__":
    main()


# screen -L -Logfile ./logs/screen_logs/laws.log python train.py --data_file "/home/dattran/datadrive2/AI-project/vietnamese_chatbot_research/input/vietnamese-chatbot/combined_law.csv" \
#   --vector_path /home/dattran/datadrive2/AI-project/vietnamese_chatbot_research/input/wiki-vi-vectors/wiki.vi.vec \
#   --epochs 6000 \
#   --save_frequency 600 \
#   --log_dir ./logs/laws
