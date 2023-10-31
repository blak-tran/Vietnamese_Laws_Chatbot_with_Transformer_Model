import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from models import BERT_Arch  # Import your custom QA model class

# Load pre-trained model and tokenizer
model_path = 'path/to/your/model.pth'  # Replace with the actual path to your saved model file
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
model = BERT_Arch.from_pretrained("vinai/phobert-base-v2")
model.load_state_dict(torch.load(model_path))
model.eval()

# List of questions for inference
questions = ["What is the capital of France?", "Who is the president of the United States?"]

# Inference function
def predict_answers(questions):
    answers = []
    for question in questions:
        inputs = tokenizer.encode_plus(question, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            start_logits, end_logits = model(input_ids, attention_mask)

        start_idx = torch.argmax(start_logits, dim=1).item()
        end_idx = torch.argmax(end_logits, dim=1).item()
        
        # Get the answer from the input tokens
        answer_tokens = input_ids[0][start_idx : end_idx + 1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        answers.append(answer)
    return answers

# Predict answers
predicted_answers = predict_answers(questions)

# Print predicted answers
for question, answer in zip(questions, predicted_answers):
    print(f"Question: {question}")
    print(f"Predicted Answer: {answer}")
    print("=" * 50)