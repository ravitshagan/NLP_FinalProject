#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import evaluate
import json
import nltk
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score
import ast
import os
from tqdm import tqdm


# In[2]:


# Disable symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Download required NLTK data
nltk.download('punkt')


# In[3]:


class DreamDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = self.tokenizer(text, 
                                   truncation=True,
                                   max_length=self.max_length,
                                   padding='max_length',
                                   return_tensors='pt')
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze()
        }


# In[4]:


def prepare_dream_dataset():
    dream_dataset = load_dataset("dream", trust_remote_code=True)
    conversations = [' '.join(item['dialogue']) for item in dream_dataset['train']]
    train_texts, test_texts = train_test_split(conversations, test_size=0.2, random_state=42)
    return train_texts, test_texts


# In[5]:


def train_model(train_dataloader, model, optimizer, device, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch+1}', leave=True):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")


# In[6]:


def generate_interpretation(dream_text, model, tokenizer, device, max_length=100):
    model.eval()
    inputs = tokenizer.encode(dream_text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            do_sample=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# In[7]:


def calculate_metrics(predictions, references):
    # BLEU Score
    bleu_scorer = nltk.translate.bleu_score
    bleu_scores = [bleu_scorer.sentence_bleu([nltk.word_tokenize(ref)], nltk.word_tokenize(pred)) for ref, pred in zip(references, predictions)]
    bleu_avg = np.mean(bleu_scores)

    # ROUGE Scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {metric: 0.0 for metric in ['rouge1', 'rouge2', 'rougeL']}

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for metric in rouge_scores:
            rouge_scores[metric] += scores[metric].fmeasure

    for metric in rouge_scores:
        rouge_scores[metric] /= len(predictions)

    # Perplexity
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    perplexity = evaluate.load("perplexity")
    perplexity_score = perplexity.compute(predictions=predictions, model_id='gpt2')['mean_perplexity']

    # BERTScore
    P, R, F1 = score(predictions, references, lang='en', verbose=True)
    bert_score = torch.mean(F1).item()

    return {
        'Average BLEU Score': bleu_avg,  # שינוי ל-Average BLEU Score
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougeL': rouge_scores['rougeL'],
        'perplexity': perplexity_score,
        'bert_score': bert_score
    }


# In[ ]:


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-7b")
    model = AutoModelForCausalLM.from_pretrained("mosaicml/mpt-7b").to(device)

    train_texts, test_texts = prepare_dream_dataset()
    train_dataset = DreamDataset(train_texts, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    train_model(train_dataloader, model, optimizer, device)

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


    predictions = [generate_interpretation(dream, model, tokenizer, device) for dream, _ in val_dreams_data]
    references = [interp for _, interp in val_dreams_data]

    metrics = calculate_metrics(predictions, references)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()


# In[ ]:




