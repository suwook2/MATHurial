import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import re

# 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'monologg/koelectra-base-v3-discriminator'
max_length = 256
batch_size = 32
num_epochs = 3

# 데이터 로드
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
submission_df = pd.read_csv('sample_submission.csv')

# 문단 분리 
paragraphs = []
for _, row in train_df.iterrows():
    text_parts = re.split(r'\n+', row['full_text'])
    for part in text_parts:
        if len(part) > 50:
            paragraphs.append({
                'text': part,
                'label': row['generated']
            })

para_df = pd.DataFrame(paragraphs)

# Train/Valid 분리
X_train, X_valid, y_train, y_valid = train_test_split(
    para_df['text'], para_df['label'], 
    test_size=0.2, random_state=42
)

# 간단한 Dataset
class SimpleDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(
                self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx], 
                dtype=torch.long
            )
        
        return item

# 모델 & 학습
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

train_dataset = SimpleDataset(X_train, y_train)
valid_dataset = SimpleDataset(X_valid, y_valid)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size*2)

# 학습
best_auc = 0
for epoch in range(num_epochs):
    # Training
    model.train()
    for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
            
            val_preds.extend(probs.cpu().numpy())
            val_labels.extend(labels.numpy())
    
    auc = roc_auc_score(val_labels, val_preds)
    print(f'Epoch {epoch+1} - Valid AUC: {auc:.4f}')
    
    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), 'best_model.pt')

# 예측
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

test_dataset = SimpleDataset(test_df['paragraph_text'])
test_loader = DataLoader(test_dataset, batch_size=batch_size*2)

predictions = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc='Predicting'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
        predictions.extend(probs.cpu().numpy())

# 후처리 - 같은 문서 스무딩
test_df['pred'] = predictions
for title in test_df['title'].unique():
    mask = test_df['title'] == title
    test_df.loc[mask, 'pred'] = 0.7 * test_df.loc[mask, 'pred'] + 0.3 * test_df.loc[mask, 'pred'].mean()

# 제출
submission_df['generated'] = test_df['pred'].values
submission_df.to_csv('submission_fast.csv', index=False)
print(f"✅ 완료! Best AUC: {best_auc:.4f}")
