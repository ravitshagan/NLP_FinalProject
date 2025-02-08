#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from sklearn.model_selection import train_test_split
import json
import pandas as pd
from tqdm import tqdm

# Load the DREAM dataset for training
def load_dream_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

test_dreams_data = {
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

# Create a custom dataset class for DREAM dataset
class DreamDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        
        for item in data:
            # Extract dialogue, question, and answer
            dialogue = ' '.join(item[0])  # Combine all dialogue turns
            qa_data = item[1][0]  # Get the first (and only) QA pair
            question = qa_data['question']
            correct_answer = qa_data['answer']
            choices = qa_data['choice']
            
            # Create input pairs for each choice
            for choice in choices:
                # Format: [CLS] dialogue [SEP] question [SEP] choice [SEP]
                text = f"{dialogue} [SEP] {question} [SEP] {choice}"
                
                encodings = tokenizer(text, truncation=True, max_length=max_length, 
                                    padding='max_length', return_tensors='pt')
                
                self.input_ids.append(encodings['input_ids'].squeeze())
                self.attn_masks.append(encodings['attention_mask'].squeeze())
                # Label is 1 if this is the correct answer, 0 otherwise
                self.labels.append(torch.tensor(1.0 if choice == correct_answer else 0.0))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attn_masks[idx],
            'labels': self.labels[idx]
        }

# Training function
def train(model, train_dataloader, val_dataloader, optimizer, scheduler, num_epochs, device):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch+1}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            model.zero_grad()
            outputs = model(input_ids,
                          attention_mask=attention_mask,
                          labels=labels.unsqueeze(1))
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
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
                              labels=labels.unsqueeze(1))
                
                loss = outputs.loss
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f'Epoch {epoch+1}:')
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}\n')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_dream_interpreter.pt')

# Function to generate dream interpretations
def generate_interpretation(dream_text, model, tokenizer, device, max_length=512):
    model.eval()
    
    # Create a dialogue-style context
    context = f"M: I had a dream last night. {dream_text}\nW: Let me help you understand what that might mean."
    question = "What could this dream symbolize?"
    
    # Predefined interpretation templates
    templates = [
        "This dream suggests anxiety about changes in your life.",
        "This dream reflects your personal growth and transformation.",
        "This dream indicates unresolved emotions or conflicts.",
        "This dream symbolizes your desires and aspirations.",
        "This dream represents your fears and uncertainties."
    ]
    
    # Score each template
    scores = []
    with torch.no_grad():
        for template in templates:
            text = f"{context} [SEP] {question} [SEP] {template}"
            inputs = tokenizer(text, return_tensors="pt", max_length=max_length,
                             truncation=True, padding='max_length').to(device)
            
            outputs = model(**inputs)
            score = outputs.logits.squeeze().item()
            scores.append(score)
    
    # Get the template with highest score and use it as base for interpretation
    best_template = templates[np.argmax(scores)]
    return best_template

# Function to evaluate model on test data
def evaluate_test_set(model, tokenizer, device, test_data):
    model.eval()
    results = []
    
    for dream, actual_interp in zip(test_data['dream'], test_data['interp']):
        generated_interp = generate_interpretation(dream, model, tokenizer, device)
        results.append({
            'dream': dream,
            'actual_interpretation': actual_interp,
            'generated_interpretation': generated_interp
        })
    
    return results

# Main execution
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the training dataset
    train_data = load_dream_dataset('dream_data.json')
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=1,  # Binary classification
        output_attentions=False,
        output_hidden_states=False
    ).to(device)
    
    # Split data into train and validation sets
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = DreamDataset(train_data, tokenizer)
    val_dataset = DreamDataset(val_data, tokenizer)
    
    # Create dataloaders
    batch_size = 8
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    
    # Create scheduler with warmup
    num_epochs = 3
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps=0,
                                              num_training_steps=total_steps)
    
    # Train the model
    train(model, train_dataloader, val_dataloader, optimizer, scheduler, num_epochs, device)
    
    # Load best model for testing
    model.load_state_dict(torch.load('best_dream_interpreter.pt'))
    
    # Test the model on the provided dreams_data
    print("\nEvaluating model on test data:")
    test_results = evaluate_test_set(model, tokenizer, device, test_dreams_data)
    
    # Print test results
    for result in test_results:
        print(f"\nDream: {result['dream']}")
        print(f"Actual Interpretation: {result['actual_interpretation']}")
        print(f"Generated Interpretation: {result['generated_interpretation']}")
        print("-" * 80)

if __name__ == "__main__":
    main()


# In[ ]:




