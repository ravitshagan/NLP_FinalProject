#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
import numpy as np
from sklearn.model_selection import train_test_split
import json
import pandas as pd
from tqdm import tqdm
import nltk
from rouge_score import rouge_scorer
from evaluate import load


# In[2]:


# Download necessary resources
nltk.download('punkt')


# In[3]:


# Load the external dataset for training
def load_dream_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# In[4]:


# Test data (provided dreams_data)
val_dreams_data = {
    "dream": [
        "Falling off a cliff", "Flying freely in the sky", "Teeth falling out one by one",
        "Being naked in a crowded street", "Failing an important exam", "Being chased by a shadowy figure",
        "Witnessing one's own death", "Running but moving in slow motion", "Meeting a deceased loved one",
        "Drowning in deep water", "Driving a car but losing control", "Walking through a dark forest",
        "Climbing a never-ending staircase", "Being trapped in a small room", "Losing one’s wallet or keys",
        "Arguing with a stranger", "Being late for an important event", "Watching a plane crash",
        "Standing in a burning building", "Falling into a deep abyss", "Seeing a baby crying",
        "Being unable to speak", "Walking barefoot on sharp rocks", "Eating spoiled food",
        "Discovering hidden treasure", "Losing hair suddenly", "Breaking a mirror",
        "Finding oneself in a strange house", "Witnessing a friend get hurt", "Flying but struggling to stay in the air",
        "Being in an endless maze", "Reuniting with a former lover", "Fighting with a family member",
        "Losing eyesight or going blind", "Seeing a snake in a dream", "Falling from a high building",
        "Receiving a gift from a stranger", "Being locked out of one’s home", "Discovering secret rooms in a familiar house",
        "Crossing a turbulent river", "Seeing oneself in a mirror", "Losing a beloved pet",
        "Sitting in an empty classroom", "Walking on thin ice", "Witnessing a sunrise",
        "Fighting a wild animal", "Seeing a collapsing building", "Being unable to find one’s way home",
        "Writing a letter that never gets sent", "Standing on a stage but forgetting lines"],
    "interp": [
        "Fear of losing control in life or anxieties about failure",
        "A deep desire for liberation from constraints or limitations",
        "Anxiety about physical appearance or communication breakdowns",
        "Feeling exposed or vulnerable in social or professional settings",
        "Fear of being judged or evaluated harshly by others",
        "Avoidance of unresolved fears or challenges in waking life",
        "Transition or significant changes happening in life",
        "A feeling of helplessness or frustration in achieving goals",
        "Processing grief or longing for past connections",
        "Overwhelmed by emotions or unconscious conflicts surfacing",
        "Concerns about control over life’s direction",
        "Navigating through uncertainty or the unconscious mind",
        "Struggling to achieve unattainable goals or self-improvement",
        "Feeling confined or stuck in a situation",
        "Anxiety about identity or security in waking life",
        "Internal conflicts or repressed emotions surfacing",
        "Pressure and fear of failing expectations",
        "Fears of failure or witnessing a significant loss",
        "Intense emotional stress or repressed anger",
        "Existential fears or fear of losing stability",
        "Anxiety about nurturing responsibilities or personal growth",
        "Feeling silenced or unheard in waking life",
        "Struggles or pain endured on the path to goals",
        "Guilt or discomfort with choices made recently",
        "A desire to uncover untapped potential or hidden talents",
        "Concerns about aging or loss of vitality",
        "Anxiety about self-image or fear of bad luck",
        "Exploring unknown aspects of the self",
        "Fear of losing someone or guilt over past actions",
        "Ambivalence about freedom or personal achievements",
        "Feeling lost or confused in waking life",
        "Nostalgia or unresolved emotions from past relationships",
        "Repressed anger or unresolved family tensions",
        "Fear of ignorance or losing perspective",
        "A symbol of transformation, fear, or temptation",
        "Anxiety about failure or public humiliation",
        "Expectation of unexpected opportunities or recognition",
        "Feeling disconnected from personal identity or safety",
        "Uncovering hidden aspects of the psyche",
        "Overcoming emotional obstacles or major life changes",
        "Reflecting on self-identity or inner conflicts",
        "Processing grief or emotional dependence",
        "Anxiety about learning or personal development",
        "Fear of taking risks or instability in life",
        "Hope and optimism for new beginnings",
        "Struggles with primal instincts or internal aggression",
        "Fear of losing stability or foundations in life",
        "Longing for security or emotional grounding",
        "Repressed communication or unspoken emotions",
        "Anxiety about performance or public judgment"]}


# In[5]:


# Create a custom dataset class for DREAM dataset
class DreamDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        
        # נניח שהנתונים כוללים רשימות של חלומות ופירושים
        dreams = data['dream']
        interpretations = data['interp']
        
        for dream, interp in zip(dreams, interpretations):
            # יצירת טקסט לצורך טוקניזציה
            text = f"Dream: {dream} Interpretation: {interp}"
            
            encodings = tokenizer(text, truncation=True, max_length=max_length, 
                                  padding='max_length', return_tensors='pt')
            
            self.input_ids.append(encodings['input_ids'].squeeze())
            self.attn_masks.append(encodings['attention_mask'].squeeze())
            self.labels.append(encodings['input_ids'].squeeze())
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attn_masks[idx],
            'labels': self.labels[idx]
        }


# In[6]:


# Modified generation function for dream interpretation
def generate_interpretation(dream_text, model, tokenizer, device, max_length=150):
    model.eval()
    # Format dream as a dialogue-style context
    context = f"M: I had a dream last night. {dream_text}\nW: Let me help you understand what that might mean."
    prompt = f"Context: {context} Question: What could this dream symbolize? Answer:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the answer part
    interpretation = generated_text.split("Answer:")[-1].strip()
    return interpretation


# In[7]:


def calculate_perplexity(predictions, tokenizer, model, device):
    log_likelihoods = []
    for pred in predictions:
        encodings = tokenizer(pred, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encodings, labels=encodings["input_ids"])
        log_likelihoods.append(outputs.loss.item())
    return np.exp(np.mean(log_likelihoods))


# In[8]:


# Training function remains the same
def train(model, train_dataloader, val_dataloader, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_train_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch+1}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            model.zero_grad()
            outputs = model(input_ids,
                          attention_mask=attention_mask,
                          labels=labels)
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids,
                              attention_mask=attention_mask,
                              labels=labels)
                
                loss = outputs.loss
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f'Epoch {epoch+1}:')
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}\n')
        
        model.train()


# In[10]:


def evaluate_model(model, tokenizer, device, dataset, dataset_name):
    model.eval()
    predictions, references = [], []
    
    for dream, actual_interp in zip(dataset['dream'], dataset['interp']):
        generated_interp = generate_interpretation(dream, model, tokenizer, device)
        predictions.append(generated_interp)
        references.append(actual_interp)
    
    # Initialize metrics
    bleu_scorer = nltk.translate.bleu_score
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    bert_score = load("bertscore")
    
    # BLEU Score
    bleu_scores = [bleu_scorer.sentence_bleu([nltk.word_tokenize(ref)], nltk.word_tokenize(pred)) for ref, pred in zip(references, predictions)]
    bleu_avg = np.mean(bleu_scores)
    
    # ROUGE Scores
    rouge1, rouge2, rougeL = 0, 0, 0
    for ref, pred in zip(references, predictions):
        scores = rouge.score(ref, pred)
        rouge1 += scores["rouge1"].fmeasure
        rouge2 += scores["rouge2"].fmeasure
        rougeL += scores["rougeL"].fmeasure
    n = len(references)
    
    # Perplexity
    perplexity = calculate_perplexity(predictions, tokenizer, model, device)
    
    # BERTScore
    bert_score_result = bert_score.compute(predictions=predictions, references=references, lang="en")
    bert_avg = np.mean(bert_score_result["f1"])
    
    # Display results
    print(f"{dataset_name} Evaluation Metrics:")
    print(f"Average BLEU Score: {bleu_avg:.4f}")
    print(f"ROUGE-1: {rouge1/n:.4f}, ROUGE-2: {rouge2/n:.4f}, ROUGE-L: {rougeL/n:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"BERTScore (F1): {bert_avg:.4f}")
    
    # Save metrics
    results = {
        "Average BLEU Score": f"{bleu_avg:.4f}",
        "ROUGE-1": f"{rouge1/n:.4f}",
        "ROUGE-2": f"{rouge2/n:.4f}",
        "ROUGE-L": f"{rougeL/n:.4f}",
        "Perplexity": f"{perplexity:.4f}",
        "BERTScore (F1)": f"{bert_avg:.4f}"
    }
    filename = f"./evaluation_metrics_{dataset_name.lower()}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation metrics saved to {filename}")


# In[ ]:


# Main execution
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the training dataset
    dream_data = load_dream_dataset('dream_data.json')
    
    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.resize_token_embeddings(len(tokenizer))
    
    # Split data into train and Test sets
    train_data, test_data = train_test_split(dream_data, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = DreamDataset(train_data, tokenizer)
    #val_dataset = DreamDataset(val_dreams_data, tokenizer)
    test_dataset = DreamDataset(test_data, tokenizer)
    
    # Create dataloaders
    batch_size = 8
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Train the model
    num_epochs = 5
    train(model, train_dataloader, val_dreams_data, optimizer, num_epochs, device)
    
    # Save the model
    model.save_pretrained('dream_interpreter_model')
    tokenizer.save_pretrained('dream_interpreter_model')
    
    # Evaluate on validation dataset
    print("\nEvaluating model on validation data:")
    evaluate_model(model, tokenizer, device, {
    "dream": [item[1][0]['question'] for item in val_data],
    "interp": [item[1][0]['answer'] for item in val_data]
    }, "Validation")

    # Evaluate on test dataset
    print("\nEvaluating model on test data:")
    evaluate_model(model, tokenizer, device, test_dreams_data, "Test")


if __name__ == "__main__":
    main()


# In[ ]:




