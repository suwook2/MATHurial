#!/usr/bin/env python3
"""
한국어 특징 추출기를 통합한 AI Text Detection 학습 파이프라인
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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

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

# korean_features.py에서 import (같은 디렉토리에 있다고 가정)
from korean_features import KoreanTextFeatureExtractor

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
    epochs = 2
    learning_rate = 2e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    gradient_accumulation_steps = 1
    
    # 특징
    use_korean_features = True
    num_features = 20  # KoreanTextFeatureExtractor가 추출하는 특징 수
    
    # 기타
    num_folds = 2
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 0

# ==================== 특징 통합 데이터셋 ====================
class FeatureEnhancedDataset(Dataset):
    def __init__(self, df, tokenizer, feature_extractor, max_length=256, is_train=True, scaler=None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.is_train = is_train
        self.scaler = scaler
        
        # 모든 텍스트에 대해 특징 추출 (시간이 걸릴 수 있음)
        print("Extracting Korean text features...")
        self.features = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Feature extraction"):
            if is_train:
                text = row['full_text']
            else:
                text = row['paragraph_text']
            
            # 한국어 특징 추출
            features = feature_extractor.extract_features(text)
            feature_vector = [
                features.get('sentence_count', 0),
                features.get('avg_sentence_length', 0),
                features.get('sentence_length_std', 0),
                features.get('style_consistency', 0),
                features.get('connective_ratio', 0),
                features.get('explanatory_ratio', 0),
                features.get('repetitive_structure', 0),
                features.get('sentence_start_diversity', 0),
                features.get('parenthesis_count', 0),
                features.get('quote_count', 0),
                features.get('bullet_points', 0),
                features.get('numbered_list', 0),
                features.get('vocabulary_diversity', 0),
                features.get('ai_expression_score', 0),
                # 추가 특징들
                len(text),  # 텍스트 길이
                text.count('.'),  # 문장 수 (간단 버전)
                text.count('습니다'),  # 격식체 빈도
                text.count('이다'),  # 평어체 빈도
                text.count('죠'),  # 구어체 빈도
                text.count('또한') + text.count('그리고'),  # 연결어 빈도
            ]
            self.features.append(feature_vector)
        
        self.features = np.array(self.features)
        
        # 특징 정규화
        if is_train and scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        elif scaler is not None:
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 텍스트 처리
        if self.is_train:
            text = row['full_text']
            label = row['generated']
        else:
            text = row['paragraph_text']
            label = -1
        
        # 토큰화
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'features': torch.tensor(self.features[idx], dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.float),
            'id': row.get('ID', idx)
        }

# ==================== 특징 통합 모델 ====================
class KoreanAwareModel(nn.Module):
    def __init__(self, model_name, num_features=20, dropout=0.1):
        super().__init__()
        # BERT 인코더
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # 수동 특징 처리
        self.feature_processor = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        
        # BERT 특징 처리
        self.bert_processor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 특징 융합
        self.fusion = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(64, 1)
        )
        
        # 게이트 메커니즘 (특징의 중요도 학습)
        self.feature_gate = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask, features):
        # BERT 인코딩
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Pooling
        cls_output = outputs.last_hidden_state[:, 0]  # [CLS] 토큰
        mean_output = outputs.last_hidden_state.mean(dim=1)  # 평균 풀링
        
        # BERT 특징과 평균 결합
        bert_features = self.bert_processor(cls_output * 0.7 + mean_output * 0.3)
        
        # 수동 특징에 게이트 적용 (중요한 특징만 선택)
        gated_features = features * self.feature_gate(features)
        manual_features = self.feature_processor(gated_features)
        
        # 특징 결합
        combined = torch.cat([bert_features, manual_features], dim=-1)
        
        # 최종 예측
        logits = self.fusion(combined)
        return logits

# ==================== 학습 함수 ====================
def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, accumulation_steps=1):
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    progress_bar = tqdm(dataloader, desc='Training')
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward
        logits = model(input_ids, attention_mask, features)
        loss = criterion(logits.squeeze(-1), labels)
        loss = loss / accumulation_steps
        
        # Backward
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # 기록
        total_loss += loss.item() * accumulation_steps
        predictions.extend(torch.sigmoid(logits).cpu().detach().numpy())
        targets.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item() * accumulation_steps})
    
    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(targets, predictions)
    
    return avg_loss, auc

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask, features)
            loss = criterion(logits.squeeze(-1), labels)
            
            total_loss += loss.item()
            predictions.extend(torch.sigmoid(logits).cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(targets, predictions)
    
    return avg_loss, auc, predictions

# ==================== 메인 학습 루프 ====================
def main():
    config = Config()
    seed_everything(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 데이터 로드
    print("Loading data...")
    train_df = pd.read_csv(config.train_path)
    test_df = pd.read_csv(config.test_path)
    
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # 한국어 특징 추출기 초기화
    print("\nInitializing Korean feature extractor...")
    feature_extractor = KoreanTextFeatureExtractor()
    
    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # K-Fold 교차 검증
    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed)
    
    oof_predictions = np.zeros(len(train_df))
    test_predictions = []
    feature_scalers = []  # 각 fold의 scaler 저장
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['generated'])):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{config.num_folds}")
        print(f"{'='*50}")
        
        # 데이터 분할
        train_data = train_df.iloc[train_idx]
        val_data = train_df.iloc[val_idx]
        
        # 데이터셋 생성 (특징 추출 포함)
        print("\nCreating datasets with feature extraction...")
        train_dataset = FeatureEnhancedDataset(
            train_data, tokenizer, feature_extractor, 
            config.max_length, is_train=True
        )
        
        # Validation 데이터셋은 train의 scaler 사용
        val_dataset = FeatureEnhancedDataset(
            val_data, tokenizer, feature_extractor, 
            config.max_length, is_train=True, 
            scaler=train_dataset.scaler
        )
        
        feature_scalers.append(train_dataset.scaler)
        
        # 데이터로더
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            num_workers=config.num_workers
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size * 2, 
            shuffle=False,
            num_workers=config.num_workers
        )
        
        # 모델 초기화
        model = KoreanAwareModel(config.model_name, config.num_features)
        model.to(config.device)
        
        # 옵티마이저
        optimizer = AdamW(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        
        num_training_steps = len(train_loader) * config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_training_steps * config.warmup_ratio),
            num_training_steps=num_training_steps
        )
        
        # 손실 함수
        criterion = nn.BCEWithLogitsLoss()
        
        # 학습
        best_auc = 0
        best_model_path = f"{config.output_dir}/best_model_fold{fold}_with_features.pt"
        
        for epoch in range(config.epochs):
            print(f"\nEpoch {epoch + 1}/{config.epochs}")
            
            # Train
            train_loss, train_auc = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, 
                config.device, config.gradient_accumulation_steps
            )
            
            # Validate
            val_loss, val_auc, val_preds = validate_epoch(
                model, val_loader, criterion, config.device
            )
            
            print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            
            # 최고 성능 모델 저장
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'scaler': train_dataset.scaler
                }, best_model_path)
                print(f"Best model saved! AUC: {best_auc:.4f}")
        
        # OOF 예측
        oof_predictions[val_idx] = np.array(val_preds).squeeze()
        
        # 메모리 정리
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
    
    # 테스트 예측 (모든 fold 모델 사용)
    print("\n" + "="*50)
    print("Predicting on test set...")
    print("="*50)
    
    for fold in range(config.num_folds):
        print(f"\nFold {fold + 1} prediction...")
        
        # 모델 로드
        checkpoint = torch.load(f"{config.output_dir}/best_model_fold{fold}_with_features.pt")
        model = KoreanAwareModel(config.model_name, config.num_features)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(config.device)
        model.eval()
        
        # 테스트 데이터셋
        test_dataset = FeatureEnhancedDataset(
            test_df, tokenizer, feature_extractor,
            config.max_length, is_train=False,
            scaler=checkpoint['scaler']
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size * 2,
            shuffle=False,
            num_workers=config.num_workers
        )
        
        # 예측
        fold_predictions = []
        ids = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f'Fold {fold+1} prediction'):
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                features = batch['features'].to(config.device)
                
                logits = model(input_ids, attention_mask, features)
                probs = torch.sigmoid(logits).cpu().numpy()
                
                fold_predictions.extend(probs)
                if fold == 0:  # ID는 첫 번째 fold에서만 수집
                    ids.extend(batch['id'].numpy())
        
        test_predictions.append(np.array(fold_predictions).squeeze())
    
    # 최종 결과
    print(f"\n{'='*50}")
    print("Final Results")
    print(f"{'='*50}")
    
    # OOF 성능
    oof_auc = roc_auc_score(train_df['generated'], oof_predictions)
    print(f"Overall OOF AUC: {oof_auc:.4f}")
    
    # 테스트 예측 평균
    test_predictions = np.mean(test_predictions, axis=0)
    
    # 문서 수준 후처리
    print("\nApplying document-level post-processing...")
    test_df['prediction'] = test_predictions
    
    # 같은 문서의 문단들 평활화
    for title in test_df['title'].unique():
        doc_mask = test_df['title'] == title
        doc_preds = test_df.loc[doc_mask, 'prediction'].values
        
        if len(doc_preds) > 1:
            # 문서 평균
            doc_mean = doc_preds.mean()
            
            # 평활화
            if doc_mean > 0.7:  # AI 문서
                smoothed = np.maximum(doc_preds, 0.5)
            elif doc_mean < 0.3:  # Human 문서
                smoothed = np.minimum(doc_preds, 0.5)
            else:
                smoothed = doc_preds * 0.8 + doc_mean * 0.2
            
            test_df.loc[doc_mask, 'prediction'] = smoothed
    
    # 제출 파일 생성
    submission = pd.DataFrame({
        'ID': ids,
        'generated': test_df['prediction'].values
    })
    submission = submission.sort_values('ID').reset_index(drop=True)
    submission.to_csv(f"{config.output_dir}/submission_with_korean_features.csv", index=False)
    print(f"\nSubmission saved to {config.output_dir}/submission_with_korean_features.csv")

if __name__ == "__main__":
    main()