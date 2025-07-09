"""
한국어 텍스트 특화 AI 탐지 특징 추출기
"""
import torch.nn as nn
import re
import numpy as np
from collections import Counter

class KoreanTextFeatureExtractor:
    def __init__(self):
        # AI가 자주 사용하는 한국어 패턴
        self.ai_patterns = {
            # 문체 전환 패턴
            'style_switch': [
                (r'이다\.', r'입니다\.'),  # 평어체 → 경어체
                (r'한다\.', r'합니다\.'),
                (r'였다\.', r'였습니다\.'),
            ],
            
            # AI가 과도하게 사용하는 연결어
            'connectives': [
                '또한', '그리고', '하지만', '그러나', '따라서',
                '이러한', '이와 같이', '즉', '예를 들어', '특히',
                '마찬가지로', '뿐만 아니라', '게다가'
            ],
            
            # 부자연스러운 존칭 사용
            'honorifics': [
                (r'예수\s', r'예수님'),
                (r'([가-힣]+)\s', r'\1님'),  # 갑작스러운 님 추가
            ],
            
            # AI 특유의 설명체
            'explanatory': [
                r'있습니다\.',
                r'것입니다\.',
                r'있죠\.',
                r'되었습니다\.',
                r'라고 할 수 있습니다\.',
            ],
            
            # 반복적 구조
            'repetitive': [
                r'첫째[,，]\s*.*둘째[,，]\s*.*셋째',  # 나열식 구조
                r'(\d+)\.\s*.*\n\s*(\d+)\.\s*',  # 번호 매기기
                r'•\s*.*\n\s*•\s*',  # 불릿 포인트
            ]
        }
    
    def extract_features(self, text):
        """텍스트에서 AI 탐지에 유용한 특징 추출"""
        features = {}
        
        # 1. 기본 통계
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        features['sentence_count'] = len(sentences)
        features['avg_sentence_length'] = np.mean([len(s) for s in sentences]) if sentences else 0
        features['sentence_length_std'] = np.std([len(s) for s in sentences]) if len(sentences) > 1 else 0
        
        # 2. 문체 일관성 점수
        style_endings = {
            'formal': 0,    # ~습니다
            'informal': 0,  # ~어요/~아요
            'plain': 0,     # ~다
            'mixed': 0      # 혼합
        }
        
        for sent in sentences:
            if re.search(r'습니다[.!?]?$', sent):
                style_endings['formal'] += 1
            elif re.search(r'[어아]요[.!?]?$', sent):
                style_endings['informal'] += 1
            elif re.search(r'다[.!?]?$', sent):
                style_endings['plain'] += 1
            else:
                style_endings['mixed'] += 1
        
        # 문체 일관성 (엔트로피가 낮을수록 일관됨)
        total = sum(style_endings.values())
        if total > 0:
            probs = [v/total for v in style_endings.values()]
            entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
            features['style_consistency'] = 1 / (1 + entropy)  # 0~1 사이 값
        else:
            features['style_consistency'] = 0
        
        # 3. 연결어 빈도
        connective_count = sum(text.count(conn) for conn in self.ai_patterns['connectives'])
        features['connective_ratio'] = connective_count / len(sentences) if sentences else 0
        
        # 4. 설명체 비율
        explanatory_count = sum(1 for pattern in self.ai_patterns['explanatory'] 
                               if re.search(pattern, text))
        features['explanatory_ratio'] = explanatory_count / len(sentences) if sentences else 0
        
        # 5. 반복 구조 검출
        repetitive_score = 0
        for pattern in self.ai_patterns['repetitive']:
            if re.search(pattern, text, re.MULTILINE):
                repetitive_score += 1
        features['repetitive_structure'] = repetitive_score
        
        # 6. 문장 시작 패턴 다양성
        sentence_starts = [sent.split()[0] if sent.split() else '' for sent in sentences]
        unique_starts = len(set(sentence_starts))
        features['sentence_start_diversity'] = unique_starts / len(sentences) if sentences else 0
        
        # 7. 괄호 사용 패턴
        features['parenthesis_count'] = text.count('(') + text.count('[')
        features['quote_count'] = text.count('"') + text.count("'")
        
        # 8. 특수 패턴 검출
        features['bullet_points'] = 1 if re.search(r'[•·▪▫◦‣⁃]', text) else 0
        features['numbered_list'] = 1 if re.search(r'^\s*\d+[.)]\s', text, re.MULTILINE) else 0
        
        # 9. 어휘 다양성
        words = re.findall(r'[가-힣]+', text)
        if words:
            unique_words = len(set(words))
            features['vocabulary_diversity'] = unique_words / len(words)
        else:
            features['vocabulary_diversity'] = 0
        
        # 10. AI 특유의 표현 점수
        ai_expression_score = 0
        
        # "~있습니다. ~있습니다." 같은 반복
        if re.search(r'있습니다\.[^.]*있습니다\.', text):
            ai_expression_score += 1
        
        # 과도한 부사 사용
        adverbs = ['매우', '정말', '아주', '특히', '상당히', '굉장히']
        adverb_count = sum(text.count(adv) for adv in adverbs)
        if adverb_count > len(sentences) * 0.3:  # 문장당 0.3개 이상
            ai_expression_score += 1
        
        features['ai_expression_score'] = ai_expression_score
        
        return features
    
    def extract_paragraph_transition_features(self, paragraphs):
        """문단 간 전환 특징 추출"""
        if len(paragraphs) < 2:
            return {}
        
        features = {}
        
        # 1. 문체 일관성
        styles = []
        for para in paragraphs:
            if re.search(r'습니다\.$', para):
                styles.append('formal')
            elif re.search(r'[어아]요\.$', para):
                styles.append('informal')
            else:
                styles.append('plain')
        
        # 문체 전환 횟수
        style_changes = sum(1 for i in range(1, len(styles)) if styles[i] != styles[i-1])
        features['style_change_ratio'] = style_changes / (len(styles) - 1)
        
        # 2. 길이 변화 패턴
        lengths = [len(para) for para in paragraphs]
        features['length_variance'] = np.var(lengths) / (np.mean(lengths) + 1)
        
        # 3. 주제 일관성 (간단한 버전)
        # 각 문단의 주요 명사 추출
        paragraph_keywords = []
        for para in paragraphs:
            nouns = re.findall(r'[가-힣]{2,}', para)
            keywords = [w for w in nouns if len(w) > 1]
            paragraph_keywords.append(set(keywords[:10]))  # 상위 10개
        
        # 인접 문단 간 키워드 중복도
        overlap_scores = []
        for i in range(1, len(paragraph_keywords)):
            if paragraph_keywords[i] and paragraph_keywords[i-1]:
                overlap = len(paragraph_keywords[i] & paragraph_keywords[i-1])
                max_size = max(len(paragraph_keywords[i]), len(paragraph_keywords[i-1]))
                overlap_scores.append(overlap / max_size)
        
        features['keyword_consistency'] = np.mean(overlap_scores) if overlap_scores else 0
        
        return features

# 특징을 모델에 통합하는 방법
class FeatureEnhancedModel(nn.Module):
    def __init__(self, transformer_model, num_features=20):
        super().__init__()
        self.transformer = transformer_model
        self.feature_projector = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        # 기존 분류기 차원 조정
        original_input_dim = 256 + 128  # StyleConsistencyModel의 차원
        self.final_classifier = nn.Sequential(
            nn.Linear(original_input_dim + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, input_ids, attention_mask, features):
        # Transformer 부분
        transformer_output = self.transformer(input_ids, attention_mask)
        
        # 수동 특징 투영
        feature_output = self.feature_projector(features)
        
        # 결합
        combined = torch.cat([transformer_output, feature_output], dim=-1)
        
        # 최종 분류
        return self.final_classifier(combined)

# 사용 예시
if __name__ == "__main__":
    extractor = KoreanTextFeatureExtractor()
    
    # 예시 텍스트
    ai_text = """이러한 상황에서 우리는 다음과 같은 점들을 고려해야 합니다. 
    첫째, 환경 보호는 매우 중요합니다. 
    둘째, 경제 발전도 중요합니다. 
    셋째, 이 둘의 균형을 맞추는 것이 필요합니다."""
    
    human_text = """어제 친구를 만났다. 오랜만에 보니 반가웠다. 
    커피를 마시며 이런저런 얘기를 나눴는데, 시간 가는 줄 몰랐다."""
    
    print("AI Text Features:")
    print(extractor.extract_features(ai_text))
    
    print("\nHuman Text Features:")
    print(extractor.extract_features(human_text))