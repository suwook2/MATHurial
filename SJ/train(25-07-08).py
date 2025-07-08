#!usrbinenv python3

AI Text Detection Model Training Pipeline
한국어 위키피디아 데이터를 사용한 AI 생성 텍스트 탐지 모델


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
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import (
    AutoModel, 
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    logging
)
logging.set_verbosity_error()

# Mixed Precision Training (메모리 절약)
try
    from torch.cuda.amp import autocast, GradScaler
    use_amp = True
except
    use_amp = False
    print(Mixed precision training not available)

# 시드 고정
def seed_everything(seed=42)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# ==================== 설정 ====================
class Config
    # 경로
    train_path = 'train.csv'
    test_path = 'test.csv'
    output_dir = '.outputs'
    
    # 모델
    model_name = 'klueroberta-base'  # 'large'는 메모리 많이 사용
    max_length = 256  # 512에서 줄임
    max_context_paragraphs = 3  # 5에서 줄임
    
    # 학습
    batch_size = 8
    epochs = 2
    learning_rate = 2e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    gradient_accumulation_steps = 2
    
    # 기타
    num_folds = 2
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 4

# ==================== 데이터셋 ====================
class AITextDataset(Dataset)
    def __init__(self, df, tokenizer, max_length=512, max_context=5, is_train=True)
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_context = max_context
        self.is_train = is_train
        
        # 문서별 그룹화
        if 'title' in df.columns
            self.doc_groups = df.groupby('title').groups
        else
            self.doc_groups = {}
    
    def __len__(self)
        return len(self.df)
    
    def __getitem__(self, idx)
        row = self.df.iloc[idx]
        
        # 학습평가 데이터에 따라 컬럼명 조정
        if self.is_train
            text = row['full_text']
            label = row['generated']
        else
            text = row['paragraph_text']
            label = -1  # 평가 데이터는 라벨 없음
        
        # 현재 텍스트 인코딩
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 문서 컨텍스트 처리 (평가 데이터만)
        context_encodings = []
        if not self.is_train and 'title' in row and pd.notna(row['title'])
            title = row['title']
            para_idx = row['paragraph_index'] if 'paragraph_index' in row else -1
            
            if title in self.doc_groups
                doc_indices = list(self.doc_groups[title])
                doc_df = self.df.loc[doc_indices].sort_values('paragraph_index')
                
                # 현재 문단 제외한 다른 문단들
                for _, para_row in doc_df.iterrows()
                    if ('paragraph_index' in para_row and para_row['paragraph_index'] != para_idx) or 'paragraph_index' not in para_row
                        para_text = para_row['paragraph_text']
                        enc = self.tokenizer(
                            para_text,
                            truncation=True,
                            padding='max_length',
                            max_length=self.max_length,
                            return_tensors='pt'
                        )
                        context_encodings.append(enc)
                        
                        if len(context_encodings) = self.max_context
                            break
        
        # 컨텍스트가 있으면 스택, 없으면 더미 생성
        if context_encodings
            context_ids = torch.cat([e['input_ids'] for e in context_encodings])
            context_mask = torch.cat([e['attention_mask'] for e in context_encodings])
        else
            # 더미 컨텍스트 (패딩된 것)
            context_ids = torch.zeros((1, self.max_length), dtype=torch.long)
            context_mask = torch.zeros((1, self.max_length), dtype=torch.long)
        
        # 고정된 크기로 패딩자르기 (배치 처리를 위해)
        if context_ids.size(0)  self.max_context
            # 패딩 추가
            padding_size = self.max_context - context_ids.size(0)
            context_ids = torch.cat([
                context_ids,
                torch.zeros((padding_size, self.max_length), dtype=torch.long)
            ])
            context_mask = torch.cat([
                context_mask,
                torch.zeros((padding_size, self.max_length), dtype=torch.long)
            ])
        else
            # 최대 개수만큼 자르기
            context_ids = context_ids[self.max_context]
            context_mask = context_mask[self.max_context]
        
        return {
            'input_ids' encoding['input_ids'].squeeze(),
            'attention_mask' encoding['attention_mask'].squeeze(),
            'context_ids' context_ids,
            'context_mask' context_mask,
            'labels' torch.tensor(label, dtype=torch.float),
            'id' row['ID'] if 'ID' in row else idx
        }

# ==================== 모델 ====================
class DocumentAwareClassifier(nn.Module)
    def __init__(self, model_name, num_labels=1, dropout=0.1)
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.config = self.encoder.config
        hidden_size = self.config.hidden_size
        
        # 문단 특징 추출
        self.paragraph_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 문서 컨텍스트 처리
        self.context_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 특징 통합
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_size  2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 최종 분류기
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size  2),
            nn.LayerNorm(hidden_size  2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size  2, num_labels)
        )
        
    def forward(self, input_ids, attention_mask, context_ids=None, context_mask=None)
        # 현재 문단 인코딩
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        para_hidden = outputs.last_hidden_state
        para_pooled = para_hidden.mean(dim=1)  # 평균 풀링
        para_features = self.paragraph_projector(para_pooled)
        
        # 컨텍스트 처리
        if context_ids is not None and context_ids.size(1)  0
            batch_size = input_ids.size(0)
            context_features = []
            
            # 각 샘플의 컨텍스트 처리
            for i in range(batch_size)
                ctx_ids = context_ids[i]
                ctx_mask = context_mask[i]
                
                # 유효한 컨텍스트만 선택
                valid_ctx = ctx_mask.sum(dim=1)  0
                if valid_ctx.any()
                    valid_ctx_ids = ctx_ids[valid_ctx]
                    valid_ctx_mask = ctx_mask[valid_ctx]
                    
                    # 컨텍스트 인코딩
                    ctx_outputs = self.encoder(
                        input_ids=valid_ctx_ids,
                        attention_mask=valid_ctx_mask
                    )
                    ctx_hidden = ctx_outputs.last_hidden_state.mean(dim=1)
                    
                    # 어텐션을 통한 컨텍스트 통합
                    para_feat_expanded = para_features[ii+1].unsqueeze(0)
                    ctx_feat_expanded = ctx_hidden.unsqueeze(0)
                    
                    attended, _ = self.context_attention(
                        para_feat_expanded,
                        ctx_feat_expanded,
                        ctx_feat_expanded
                    )
                    context_features.append(attended.squeeze(0))
                else
                    # 컨텍스트가 없으면 제로 벡터
                    context_features.append(torch.zeros_like(para_features[ii+1]))
            
            context_features = torch.cat(context_features, dim=0)
        else
            # 컨텍스트가 없으면 제로 벡터
            context_features = torch.zeros_like(para_features)
        
        # 특징 결합
        combined_features = torch.cat([para_features, context_features], dim=-1)
        fused_features = self.feature_fusion(combined_features)
        
        # 최종 분류
        logits = self.classifier(fused_features)
        return logits

# ==================== 학습 함수 ====================
def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, accumulation_steps=1)
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    progress_bar = tqdm(dataloader, desc='Training')
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        context_ids = batch['context_ids'].to(device)
        context_mask = batch['context_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward
        logits = model(input_ids, attention_mask, context_ids, context_mask)
        loss = criterion(logits.squeeze(-1), labels)
        loss = loss  accumulation_steps
        
        # Backward
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # 기록
        total_loss += loss.item()  accumulation_steps
        predictions.extend(torch.sigmoid(logits).cpu().detach().numpy())
        targets.extend(labels.cpu().numpy())
        
        # 진행상황 업데이트
        progress_bar.set_postfix({'loss' loss.item()  accumulation_steps})
    
    avg_loss = total_loss  len(dataloader)
    auc = roc_auc_score(targets, predictions)
    
    return avg_loss, auc

def validate_epoch(model, dataloader, criterion, device)
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad()
        for batch in tqdm(dataloader, desc='Validating')
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            context_ids = batch['context_ids'].to(device)
            context_mask = batch['context_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask, context_ids, context_mask)
            loss = criterion(logits.squeeze(-1), labels)
            
            total_loss += loss.item()
            predictions.extend(torch.sigmoid(logits).cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    avg_loss = total_loss  len(dataloader)
    auc = roc_auc_score(targets, predictions)
    
    return avg_loss, auc, predictions

# ==================== 추론 함수 ====================
def predict(model, dataloader, device)
    model.eval()
    predictions = []
    ids = []
    
    with torch.no_grad()
        for batch in tqdm(dataloader, desc='Predicting')
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            context_ids = batch['context_ids'].to(device)
            context_mask = batch['context_mask'].to(device)
            batch_ids = batch['id']
            
            logits = model(input_ids, attention_mask, context_ids, context_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            predictions.extend(probs)
            ids.extend(batch_ids.numpy() if torch.is_tensor(batch_ids) else batch_ids)
    
    return ids, predictions

# ==================== 메인 학습 루프 ====================
def main()
    # 설정
    config = Config()
    seed_everything(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 데이터 로드
    print(Loading data...)
    
    # 파일 존재 확인
    if not os.path.exists(config.train_path)
        raise FileNotFoundError(fTraining file not found {config.train_path})
    if not os.path.exists(config.test_path)
        raise FileNotFoundError(fTest file not found {config.test_path})
    
    train_df = pd.read_csv(config.train_path)
    test_df = pd.read_csv(config.test_path)
    
    print(fTrain samples {len(train_df)})
    print(fTest samples {len(test_df)})
    print(fClass distributionn{train_df['generated'].value_counts()})
    print(fUsing device {config.device})
    
    # 컬럼 확인
    print(fnTrain columns {list(train_df.columns)})
    print(fTest columns {list(test_df.columns)})
    
    # GPU 정보 출력
    if torch.cuda.is_available()
        print(fGPU {torch.cuda.get_device_name(0)})
        print(fGPU Memory {torch.cuda.get_device_properties(0).total_memory  10243.1f} GB)
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # K-Fold 교차 검증
    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed)
    
    oof_predictions = np.zeros(len(train_df))
    test_predictions = []
    test_ids = None  # 한 번만 저장
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['generated']))
        print(fn{'='50})
        print(fFold {fold + 1}{config.num_folds})
        print(f{'='50})
        
        # 데이터 분할
        train_data = train_df.iloc[train_idx]
        val_data = train_df.iloc[val_idx]
        
        # 데이터셋 생성
        train_dataset = AITextDataset(train_data, tokenizer, config.max_length, is_train=True)
        val_dataset = AITextDataset(val_data, tokenizer, config.max_length, is_train=True)
        test_dataset = AITextDataset(test_df, tokenizer, config.max_length, config.max_context_paragraphs, is_train=False)
        
        # 데이터로더
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size  2, 
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.batch_size  2, 
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # 모델 초기화
        model = DocumentAwareClassifier(config.model_name)
        model.to(config.device)
        
        # 모델 파라미터 수 출력
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(fnModel parameters {total_params1e6.1f}M (trainable {trainable_params1e6.1f}M))
        
        # 옵티마이저 및 스케줄러
        optimizer = AdamW(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        
        num_training_steps = len(train_loader)  config.epochs  config.gradient_accumulation_steps
        num_warmup_steps = int(num_training_steps  config.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # 손실 함수
        criterion = nn.BCEWithLogitsLoss()
        
        # 학습
        best_auc = 0
        best_model_path = f{config.output_dir}best_model_fold{fold}.pt
        best_val_preds = None  # 최고 성능일 때의 예측값 저장
        
        for epoch in range(config.epochs)
            print(fnEpoch {epoch + 1}{config.epochs})
            
            # Train
            train_loss, train_auc = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, 
                config.device, config.gradient_accumulation_steps
            )
            
            # Validate
            val_loss, val_auc, val_preds = validate_epoch(
                model, val_loader, criterion, config.device
            )
            
            print(fTrain Loss {train_loss.4f}, Train AUC {train_auc.4f})
            print(fVal Loss {val_loss.4f}, Val AUC {val_auc.4f})
            
            # 최고 성능 모델 저장
            if val_auc  best_auc
                best_auc = val_auc
                best_val_preds = val_preds  # 최고 성능일 때의 예측값 저장
                torch.save(model.state_dict(), best_model_path)
                print(fBest model saved! AUC {best_auc.4f})
        
        # OOF 예측 (이미 저장된 best_val_preds 사용)
        if best_val_preds is not None
            oof_predictions[val_idx] = np.array(best_val_preds).squeeze()
        else
            # epoch이 1개인 경우를 위한 처리
            oof_predictions[val_idx] = np.array(val_preds).squeeze()
        
        # 테스트 예측
        fold_test_ids, fold_test_preds = predict(model, test_loader, config.device)
        if test_ids is None  # 첫 번째 fold에서만 ID 저장
            test_ids = fold_test_ids
        test_predictions.append(np.array(fold_test_preds).squeeze())
        
        # 메모리 정리
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
    
    # 최종 결과
    print(fn{'='50})
    print(Final Results)
    print(f{'='50})
    
    # OOF 성능
    oof_auc = roc_auc_score(train_df['generated'], oof_predictions)
    print(fOverall OOF AUC {oof_auc.4f})
    
    # 테스트 예측 평균
    test_predictions = np.mean(test_predictions, axis=0)
    
    # 제출 파일 생성
    submission = pd.DataFrame({
        'ID' test_ids,
        'generated' test_predictions
    })
    submission = submission.sort_values('ID').reset_index(drop=True)
    submission.to_csv(f{config.output_dir}submission.csv, index=False)
    print(fnSubmission saved to {config.output_dir}submission.csv)
    
    # OOF 예측 저장 (추가 분석용)
    oof_df = train_df.copy()
    oof_df['predicted'] = oof_predictions
    oof_df.to_csv(f{config.output_dir}oof_predictions.csv, index=False)
    print(fOOF predictions saved to {config.output_dir}oof_predictions.csv)

if __name__ == __main__
    main()