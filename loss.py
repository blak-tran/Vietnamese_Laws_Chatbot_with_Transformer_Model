import tensorflow as tf
import numpy as np 
from nltk.translate.bleu_score import sentence_bleu
import torch.nn.functional as F
import torch

epsilon = 1e-7

@tf.function(reduce_retracing=True)
def loss_function(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, 44))
  
  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')(y_true, y_pred)

  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
  loss = tf.multiply(loss, mask)

  return tf.reduce_mean(loss)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=2000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)  # Cast d_model to float32
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)  # Cast step to float32
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
  
def cal_avg_bleu(target_tokens, predicted_tokens):
  bleu_scores = []
  for i in range(len(predicted_tokens)):
      bleu = sentence_bleu([target_tokens[i]], predicted_tokens[i]) 
      bleu_scores.append(bleu)
      
  avg_bleu = np.mean(bleu_scores) 
  return avg_bleu

def cal_f1_score(target_tokens, predicted_ids):
 predicted_ids = predicted_ids.numpy()
 target_tokens = target_tokens.numpy()
  
 tp = 0 # True positives
 fp = 0 # False positives 
 fn = 0 # False negatives
  
 for target, pred in zip(target_tokens, predicted_ids):
    
    target_set = set(target)
    pred_set = set(pred)
    
    # True positives are common elements
    tp += len(target_set & pred_set)  

    # False positives are predicted but not in target
    fp += len(pred_set - target_set)

    # False negatives are target but not predicted
    fn += len(target_set - pred_set)
 if len(predicted_ids) == 0:
    return 0

 precision = tp / (tp + fp + epsilon)
 recall = tp / (tp + fn + epsilon)

 if precision + recall == 0:
  return 0
 else:
  f1 = 2 * precision * recall / (precision + recall + epsilon)  

 return f1

def cal_loss_bert(start_logits, start_positions, end_logits, end_positions, criterion):
  start_losses = []
  end_losses = []


  for i in range(start_logits.shape[0]):
    start_idx = start_positions[i].argmax()
    
    start_loss = criterion(start_logits[i], start_idx)

    end_idx = end_positions[i].argmax()
    
    end_loss = criterion(end_logits[i], end_idx)

    start_losses.append(start_loss)
    end_losses.append(end_loss)

  start_loss = torch.stack(start_losses).mean()  
  end_loss = torch.stack(end_losses).mean()
  
  return start_loss + end_loss