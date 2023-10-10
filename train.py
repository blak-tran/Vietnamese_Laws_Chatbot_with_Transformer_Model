import argparse
import logging
import os, pytz
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
from datetime import datetime

import pandas as pd
import tensorflow as tf
from preprocess_data import data_processing
from models import transformer
from loss import CustomSchedule, loss_function, accuracy
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Training script for chatbot model.')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--embedding_dim', type=int, default=300, help='Dimensionality of word embeddings')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save training logs')
    parser.add_argument('--data_file', type=str, default='./input/vietnamese-chatbot/vi-QA.csv', help='Path to CSV file containing training data')
    parser.add_argument('--save_frequency', type=int, default=500, help="Model checkpoint will be saved every epochs th")
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the pre-trained model checkpoint')
    parser.add_argument('--vector_path', type=str, default="", help='Path to vector file')
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--units', type=int, default=512)
    
    
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

def train_model(args, encoder_input_data, decoder_input_data, decoder_output_data, maxlen_answers, embedding_matrix, maxlen_questions, VOCAB_SIZE, embeddings_dim):
    tf.keras.backend.clear_session()
    strategy = setup_tpu()

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

    learning_rate = CustomSchedule(embeddings_dim)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

    # Set up TensorBoard logging
    tensorboard_callback = TensorBoard(log_dir=args.log_dir, histogram_freq=1)

    # Set up ModelCheckpoint callback to save model weights
    checkpoint_callback = ModelCheckpoint(
        os.path.join(args.save_dir, "checkpoint_{epoch}.h5"),
        save_freq=args.save_frequency,
        save_weights_only=True,
        verbose=1
    )
    
    if args.checkpoint_path is not None:
        # Load the checkpoint weights
        model.load_weights(args.checkpoint_path)
        print(f"Loaded checkpoint weights from {args.checkpoint_path}")

    # Training the model with TensorBoard and ModelCheckpoint callbacks
    history = model.fit([encoder_input_data, decoder_input_data], decoder_output_data, batch_size=args.batch_size, epochs=args.epochs, callbacks=[tensorboard_callback, checkpoint_callback])

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
    
    logging.basicConfig(filename=f'{args.log_dir}/{current_datetime}_training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    strategy = setup_tpu()
    dataframe = pd.read_csv(args.data_file)
    
    encoder_input_data, decoder_input_data, decoder_output_data, maxlen_answers, embedding_matrix, maxlen_questions, VOCAB_SIZE, embeddings_dim = data_processing(dataframe=dataframe, vector_path=args.vector_path)
    
    train_model(args, encoder_input_data, decoder_input_data, decoder_output_data, maxlen_answers, embedding_matrix, maxlen_questions, VOCAB_SIZE, embeddings_dim)

if __name__ == "__main__":
    main()
