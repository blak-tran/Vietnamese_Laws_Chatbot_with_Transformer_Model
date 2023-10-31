import argparse
import os, pytz
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import pandas as pd
from preprocess_data import data_processing
from preprocess_data import QADataset
from torch.utils.data import DataLoader
import torch
from torch import nn
from transformers import AdamW
from models import BERT_Arch
from tqdm import tqdm
from loss import cal_loss_bert

from sklearn.model_selection import train_test_split
from transformers import default_data_collator, AutoModelForQuestionAnswering, RobertaForQuestionAnswering, BertForQuestionAnswering
from transformers import logging

logging.set_verbosity_warning()
def parse_args():
    parser = argparse.ArgumentParser(description='Training script for chatbot model.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of model layers')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--embedding_dim', type=int, default=300, help='Dimensionality of word embeddings')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save training logs')
    parser.add_argument('--data_file', type=str, default='./input/vietnamese-chatbot/vi-QA.csv', help='Path to CSV file containing training data')
    parser.add_argument('--save_frequency', type=int, default=2, help="Model checkpoint will be saved every epochs th")
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the pre-trained model checkpoint')
    parser.add_argument('--vector_path', type=str, default="", help='Path to vector file')
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--units', type=int, default=128)
    parser.add_argument('--config', type=str, default=None)
    
    
    return parser.parse_args()


def train_model(train_dataloader, val_dataloader, model, optimizer, criterion, device, epochs, eval_every):
    model.train()  # Set the model to training mode    
    with torch.autograd.detect_anomaly():
        # Setup LR scheduler 
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        for epoch in range(epochs):
            total_loss = 0
            model_qa.train()
            loop = tqdm(train_dataloader, leave=True)
            for batch in loop:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(torch.float).to(device)
                end_positions = batch['end_positions'].to(torch.float).to(device)

                # Forward pass
                preds = model_qa(input_ids, attention_mask=attention_mask)
                start_logits = preds['start_logits']
                end_logits = preds['end_logits']
 
                # Compute the loss
                loss = cal_loss_bert(start_logits, start_positions, end_logits, end_positions, criterion)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                loop.set_description(f'Epoch {epoch+1}')
                loop.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(train_dataloader)
            print(f'Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}')
            
            # Validate every N epochs
            if epoch % eval_every == 0:
                # Switch model to evaluation mode
                model_qa.eval()  

                total_val_loss = 0
                val_loop = tqdm(val_dataloader, leave=True)
                
                # Disable gradients for validation
                with torch.no_grad():
                    for val_batch in val_loop:
                        val_input_ids = val_batch['input_ids'].to(device)
                        val_attention_mask = val_batch['attention_mask'].to(device)
                        val_start_positions = val_batch['start_positions'].to(torch.float).to(device)
                        val_end_positions = val_batch['end_positions'].to(torch.float).to(device)

                        # Forward pass
                        preds = model_qa(val_input_ids, attention_mask=val_attention_mask)
                        val_start_logits = preds['start_logits']
                        val_end_logits = preds['end_logits']
        
                        # Compute the loss
                        val_loss = cal_loss_bert(val_start_logits, val_start_positions, val_end_logits, val_end_positions, criterion)
                        
                        total_val_loss += val_loss.item()
                        val_loop.set_description(f'Val_Epoch {epoch+1}')
                        val_loop.set_postfix(val_loss=val_loss.item())

                    avg_total_val_loss = total_val_loss / len(val_dataloader)
                    lr_scheduler.step(avg_total_val_loss)
                    
                    print(f'Val_Epoch {epoch+1}/{args.epochs}, Val_loss: {avg_total_val_loss:.4f}')
                

                    checkpoint_name = f'model_epoch_{epoch}.pt'
                    
                    # Save model state_dict
                    torch.save(model.state_dict(), checkpoint_name)
                    
                    print(f'Checkpoint {checkpoint_name} saved!')

    return model
   

if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    torch.cuda.empty_cache()
    
    dataframe = pd.read_csv(args.data_file)
    # Load data
    questions, answers = data_processing(dataframe)
    train_questions, test_questions, train_answers, test_answers = train_test_split(questions, answers, test_size=0.2)
    train_questions, val_questions, train_answers, val_answers = train_test_split(train_questions, train_answers, test_size=0.2)

    # Create datasets and dataloaders
    train_dataset = QADataset(train_questions, train_answers) 
    val_dataset = QADataset(val_questions, val_answers)
    test_dataset = QADataset(test_questions, test_answers)

    collate_fn = default_data_collator
    train_dataloader = DataLoader(train_dataset, batch_size=48, shuffle=True, collate_fn=collate_fn)  
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # Load the pre-trained model with specific output layers for question answering
    model_name = "vinai/phobert-base-v2"
    model = RobertaForQuestionAnswering.from_pretrained(model_name)

    # Load the pre-trained weights for the specific output layers
    model_qa = RobertaForQuestionAnswering.from_pretrained(model_name, state_dict=model.state_dict())

    # model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

    # Push the model to GPU
    model_qa = model.to(device)

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    epochs = args.epochs  # Number of epochs for training
    eval_every = args.save_frequency
    trained_model = train_model(train_dataloader, val_dataloader, model_qa, optimizer, criterion, device, epochs, eval_every)


