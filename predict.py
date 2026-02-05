import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import re
import warnings
warnings.filterwarnings('ignore')

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")

# =====================================
# 1. 전처리 
# =====================================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'\s+', ' ', str(text))
    text = text.strip()
    return text

# =====================================
# 2. KoELECTRA 모델 불러오기
# =====================================
# KoELECTRA 모델 경로들 (우선순위 순)
model_paths = [
    './ai_detection_koelectra_model_0.95/best_koelectra_soft_cleaning',
    './ai_detection_koelectra_model_0.95',
    './ai_detection_koelectra_model_0.95/checkpoint-latest',
]

model_loaded = False
model_info = None

for model_path in model_paths:
    try:
        print(f"모델 로딩 시도 중: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        
        model_info = {
            'model': model,
            'tokenizer': tokenizer,
            'path': model_path,
            'name': 'koelectra-base-v3',
            'type': 'koelectra'
        }
        
        print(f"✅ KoELECTRA 모델 로딩 완료 (경로: {model_path})")
        model_loaded = True
        break
        
    except Exception as e:
        print(f"❌ {model_path} 로딩 실패: {e}")
        continue

if not model_loaded:
    print("❌ 모델 로딩 실패 - KoELECTRA 모델을 찾을 수 없습니다.")
    print("학습된 모델이 다음 경로에 있는지 확인해주세요:")
    for path in model_paths:
        print(f"  - {path}")
    exit()

print(f"\n✅ KoELECTRA 모델 로딩 완료")

# =====================================
# 3. 테스트 데이터 로드 및 전처리
# =====================================
test_df = pd.read_csv('./test.csv', encoding='utf-8-sig')
test_df['paragraph_text'] = test_df['paragraph_text'].apply(clean_text)
print(f"테스트 데이터 크기: {test_df.shape}")

# =====================================
# 4. 배치 예측 함수
# =====================================
def predict_batch(texts, model, tokenizer, batch_size=32, max_length=512):
    all_probs = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="배치 예측"):
        batch_texts = texts[i:i+batch_size]
        
        # 배치 토큰화
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # 디바이스로 이동
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # 배치 예측
        with torch.no_grad():
            outputs = model(**encoded)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
    
    return np.array(all_probs)

# =====================================
# 5. KoELECTRA 모델로 예측 수행
# =====================================
print("\nKoELECTRA 모델 예측 시작...")

# 배치 예측 수행
predictions = predict_batch(
    test_df['paragraph_text'].tolist(),
    model_info['model'],
    model_info['tokenizer'],
    batch_size=16,  # 메모리에 따라 조정
    max_length=512
)

print(f"✅ KoELECTRA 예측 완료 - 평균 예측값: {predictions.mean():.4f}")

# =====================================
# 6. 결과 저장
# =====================================
print("\n결과 저장 중...")

# 결과 DataFrame 생성
result_df = pd.DataFrame({
    'ID': test_df['ID'],
    'generated': predictions
})

# 결과 파일 저장
output_filename = './submission_koelectra_base_v3.csv'
result_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"✅ {output_filename} 저장 완료")

# =====================================
# 7. 결과 요약
# =====================================
print(f"\n🎯 예측 완료!")
print(f"\n📋 생성된 파일:")
print(f"- {output_filename}")

print(f"\n📊 예측 결과 통계:")
print(f"   - 평균: {predictions.mean():.4f}")
print(f"   - 표준편차: {predictions.std():.4f}")
print(f"   - 범위: {predictions.min():.4f} ~ {predictions.max():.4f}")
print(f"   - 0.0~0.3: {(predictions < 0.3).sum()}개")
print(f"   - 0.3~0.7: {((predictions >= 0.3) & (predictions < 0.7)).sum()}개")
print(f"   - 0.7~1.0: {(predictions >= 0.7).sum()}개")

print(f"\n🔍 모델 정보:")
print(f"   - 모델: {model_info['name']}")
print(f"   - 경로: {model_info['path']}")
print(f"   - 디바이스: {device}")

print(f"\n💾 결과 파일 경로: {output_filename}")
print(f"   - 총 예측 샘플 수: {len(predictions)}")
print(f"   - 파일 형식: CSV (ID, generated)")

print(f"\n🎉 추론 작업 완료!")
