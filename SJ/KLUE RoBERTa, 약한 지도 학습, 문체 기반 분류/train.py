#!/usr/bin/env python3
"""
개선된 AI Text Detection Model Training Pipeline
문단 단위 학습 + Weak Supervision 적용
"""

import os
import re
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoModel, 
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    logging
)
logging.set_verbosity_error()

# 시드 고정
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# ==================== 설정 ====================
class Config:
    # 경로
    train_path = 'train.csv'
    test_path = 'test.csv'
    output_dir = './outputs'
    
    # 모델
    model_name = 'klue/roberta-base'
    max_length = 256
    
    # 학습
    batch_size = 16
    epochs = 3
    learning_rate = 2e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    
    # 문단 분할
    paragraph_stride = 128  # 문단 분할 시 오버랩
    min_paragraph_length = 50  # 최소 문단 길이
    
    # 기타
    num_folds = 5
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 0

# ==================== 문단 분할 함수 ====================
def split_text_to_paragraphs(text, min_length=50):
    """텍스트를 문단으로 분할"""
    # 1. 우선 \n\n으로 분할
    paragraphs = text.split('\n\n')
    
    # 2. 각 문단이 너무 길면 문장 단위로 추가 분할
    final_paragraphs = []
    for para in paragraphs:
        if len(para) < min_length:
            continue
            
        # 문장으로 분할
        sentences = re.split(r'(?<=[.!?])\s+', para)
        
        # 문장들을 적절한 크기로 묶기
        current_para = ""
        for sent in sentences:
            if len(current_para) + len(sent) < 500:  # 적절한 크기
                current_para += sent + " "
            else:
                if current_para:
                    final_paragraphs.append(current_para.strip())
                current_para = sent + " "
        
        if current_para and len(current_para) > min_length:
            final_paragraphs.append(current_para.strip())
    
    return final_paragraphs if final_paragraphs else [text]

# ==================== 개선된 데이터셋 ====================
class ImprovedAITextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256, is_train=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        
        if is_train:
            # 학습 데이터: 문단으로 분할
            self.data = []
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Splitting paragraphs"):
                text = row['full_text']
                label = row['generated']
                title = row.get('title', f'doc_{idx}')
                
                # 텍스트를 문단으로 분할
                paragraphs = split_text_to_paragraphs(text)
                
                for para_idx, para in enumerate(paragraphs):
                    self.data.append({
                        'text': para,
                        'label': label,
                        'title': title,
                        'paragraph_index': para_idx,
                        'doc_label': label,  # 원본 문서 라벨
                        'num_paragraphs': len(paragraphs)
                    })
        else:
            # 평가 데이터: 그대로 사용
            self.data = []
            for idx, row in df.iterrows():
                self.data.append({
                    'text': row['paragraph_text'],
                    'title': row['title'],
                    'paragraph_index': row['paragraph_index'],
                    'id': row['ID']
                })
        
        # 문서별 그룹화 (컨텍스트 활용용)
        self.doc_groups = {}
        for idx, item in enumerate(self.data):
            title = item['title']
            if title not in self.doc_groups:
                self.doc_groups[title] = []
            self.doc_groups[title].append(idx)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # 텍스트 인코딩
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }
        
        if self.is_train:
            # Weak label 생성
            doc_label = item['doc_label']
            para_idx = item['paragraph_index']
            num_paras = item['num_paragraphs']
            
            # 휴리스틱: AI 문서의 경우 뒷부분 문단이 AI일 확률이 높음
            if doc_label == 1:
                # 문서가 AI인 경우: 뒤로 갈수록 AI 확률 증가
                weak_label = 0.3 + 0.7 * (para_idx / max(num_paras - 1, 1))
            else:
                # 문서가 Human인 경우: 모든 문단이 Human일 확률 높음
                weak_label = 0.1  # 약간의 노이즈
            
            result['labels'] = torch.tensor(weak_label, dtype=torch.float)
            result['hard_label'] = torch.tensor(doc_label, dtype=torch.float)
        else:
            result['id'] = item['id']
            
        return result

# ==================== 개선된 모델 ====================
class StyleConsistencyModel(nn.Module):
    """문체 일관성을 학습하는 모델"""
    def __init__(self, model_name, num_labels=1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # 문체 특징 추출기
        self.style_extractor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 언어적 특징 추출기
        self.linguistic_features = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 최종 분류기
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_labels)
        )
        
    def forward(self, input_ids, attention_mask):
        # BERT 인코딩
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Pooling: [CLS] 토큰 + 평균 풀링
        cls_output = outputs.last_hidden_state[:, 0]  # [CLS]
        mean_output = outputs.last_hidden_state.mean(dim=1)  # 평균
        
        # 문체 특징
        style_features = self.style_extractor(cls_output)
        
        # 언어적 특징
        linguistic_features = self.linguistic_features(mean_output)
        
        # 결합
        combined = torch.cat([style_features, linguistic_features], dim=-1)
        
        # 분류
        logits = self.classifier(combined)
        return logits

# ==================== 손실 함수 ====================
class WeakSupervisionLoss(nn.Module):
    """Weak supervision을 위한 손실 함수"""
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, logits, weak_labels, hard_labels=None):
        probs = torch.sigmoid(logits.squeeze())
        
        # Weak label loss (기본)
        weak_loss = F.binary_cross_entropy(probs, weak_labels)
        
        if hard_labels is not None:
            # Hard label loss (문서 수준)
            # 같은 문서의 문단들의 평균이 hard label에 가까워지도록
            hard_loss = F.binary_cross_entropy(probs, hard_labels)
            
            return self.alpha * weak_loss + (1 - self.alpha) * hard_loss
        
        return weak_loss

# ==================== 학습 함수 ====================
def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        hard_labels = batch.get('hard_label', labels).to(device)
        
        # Forward
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        
        # Loss
        loss = criterion(logits, labels, hard_labels)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # 기록
        total_loss += loss.item()
        predictions.extend(torch.sigmoid(logits).cpu().detach().numpy())
        targets.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    
    # Hard label 기준으로 AUC 계산 (더 의미있음)
    return avg_loss

def validate_epoch(model, dataloader, device):
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            
            predictions.extend(torch.sigmoid(logits).cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    predictions = np.array(predictions).squeeze()
    targets = np.array(targets)
    
    # Weak label 기준 AUC
    auc = roc_auc_score(targets, predictions)
    
    return auc, predictions

# ==================== 메인 학습 루프 ====================
def main():
    config = Config()
    seed_everything(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 데이터 로드
    print("Loading data...")
    train_df = pd.read_csv(config.train_path)
    test_df = pd.read_csv(config.test_path)
    
    print(f"Train documents: {len(train_df)}")
    print(f"Test paragraphs: {len(test_df)}")
    print(f"Class distribution:\n{train_df['generated'].value_counts()}")
    
    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # 전체 학습 (간단히)
    print("\nCreating paragraph dataset...")
    full_dataset = ImprovedAITextDataset(train_df, tokenizer, config.max_length, is_train=True)
    test_dataset = ImprovedAITextDataset(test_df, tokenizer, config.max_length, is_train=False)
    
    print(f"Total training paragraphs: {len(full_dataset)}")
    
    # 데이터로더
    train_loader = DataLoader(
        full_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # 모델 초기화
    print("\nInitializing model...")
    model = StyleConsistencyModel(config.model_name)
    model.to(config.device)
    
    # 옵티마이저
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    num_training_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * config.warmup_ratio),
        num_training_steps=num_training_steps
    )
    
    # 손실 함수
    criterion = WeakSupervisionLoss(alpha=0.7)
    
    # 학습
    print("\nTraining...")
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, config.device)
        print(f"Train Loss: {train_loss:.4f}")
    
    # 예측
    print("\nPredicting...")
    model.eval()
    predictions = []
    ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Predicting'):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            predictions.extend(probs)
            ids.extend(batch['id'].numpy())
    
    # 문서 수준 후처리
    print("\nPost-processing predictions...")
    test_df['prediction'] = predictions
    
    # 같은 문서 내 문단들의 예측값 평활화
    for title in test_df['title'].unique():
        doc_mask = test_df['title'] == title
        doc_preds = test_df.loc[doc_mask, 'prediction'].values
        
        if len(doc_preds) > 1:
            # 평활화: 극단적인 값 조정
            smoothed = doc_preds.copy()
            
            # 문서 내 일관성 강화
            mean_pred = doc_preds.mean()
            if mean_pred > 0.7:  # AI 문서로 판단
                smoothed = np.maximum(smoothed, 0.5)  # 최소 0.5
            elif mean_pred < 0.3:  # Human 문서로 판단
                smoothed = np.minimum(smoothed, 0.5)  # 최대 0.5
            
            # 부드러운 전환
            smoothed = 0.7 * smoothed + 0.3 * mean_pred
            
            test_df.loc[doc_mask, 'prediction'] = smoothed
    
    # 제출 파일 생성
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'generated': test_df['prediction']
    })
    submission = submission.sort_values('ID').reset_index(drop=True)
    submission.to_csv(f"{config.output_dir}/submission_improved.csv", index=False)
    print(f"\nSubmission saved to {config.output_dir}/submission_improved.csv")

if __name__ == "__main__":
    main()