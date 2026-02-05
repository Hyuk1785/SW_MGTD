import pandas as pd
import torch
import numpy as np
import kss
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from sklearn.metrics import roc_auc_score, classification_report
import re
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

df = pd.read_csv('./train_paragraphs.csv', encoding='utf-8-sig')

print(f"데이터 형태: {df.shape}")
print(f"컬럼: {df.columns.tolist()}")
print(f"라벨 분포:\n{df['generated'].value_counts()}")

def clean_text(text):
if pd.isna(text):
return ""
text = re.sub(r'\s+', ' ', str(text))
text = text.strip()
return text

df['paragraph_text'] = df['paragraph_text'].apply(clean_text)

df_1 = df[df['generated'] == 1].sample(20000, random_state=42)
print(f"Generated=1 샘플 수: {len(df_1)}")

df_0 = df[df['generated'] == 0].sample(20000, random_state=42)
print(f"Generated=0 샘플 수: {len(df_0)}")

df_balanced = pd.concat([df_0, df_1]).sample(frac=1, random_state=42).reset_index(drop=True)

model_configs = [
{
'name': 'monologg/koelectra-base-v3-discriminator',
'max_length': 512,
'learning_rate': 2e-5,        # KoELECTRA에 적합한 학습률
'batch_size': 4,              # Base 모델이므로 적당한 배치 크기
'model_type': 'koelectra'
}
]

models_and_predictions = []

for config in model_configs:
print(f"\n{'='*50}")
print(f"모델 학습: {config['name']}")
print(f"{'='*50}")

```
tokenizer = AutoTokenizer.from_pretrained(config['name'])

# 토크나이저에 pad_token이 없는 경우 추가
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# max_length 초과 샘플 제거
df_balanced = df_balanced[df_balanced['paragraph_text'].apply(lambda x: len(tokenizer.encode(str(x), add_special_tokens=True)) <= config['max_length'])].reset_index(drop=True)

model = AutoModelForSequenceClassification.from_pretrained(config['name'], num_labels=2)

def preprocess_with_features(example):
    return tokenizer(
        example['paragraph_text'],
        truncation=True,
        padding='max_length',
        max_length=config['max_length']
    )

# 데이터셋을 train/validation으로 분할
dataset = Dataset.from_pandas(df_balanced[['paragraph_text', 'generated']])
dataset = dataset.rename_column('generated', 'labels')

# ClassLabel로 변환 (stratified split을 위해 필요)
dataset = dataset.class_encode_column('labels')

# 80:20 비율로 train/val 분할
split_dataset = dataset.train_test_split(test_size=0.2, stratify_by_column='labels', seed=42)
train_dataset = split_dataset['train']
val_dataset = split_dataset['test']

# 토크나이징은 분할 후에 수행
train_dataset = train_dataset.map(preprocess_with_features, batched=True)
val_dataset = val_dataset.map(preprocess_with_features, batched=True)

print(f"학습 데이터: {len(train_dataset)}개")
print(f"검증 데이터: {len(val_dataset)}개")

model_output_dir = f'./ai_detection_{config["model_type"]}_model_0.95'

training_args = TrainingArguments(
    output_dir=model_output_dir,
    save_strategy="epoch",
    eval_strategy="epoch",  # 매 에포크마다 검증
    learning_rate=config['learning_rate'],
    per_device_train_batch_size=config['batch_size'],
    per_device_eval_batch_size=config['batch_size'] * 2,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,  # 최고 성능 모델 로드
    metric_for_best_model="eval_loss",  # 최저 val loss 기준
    greater_is_better=False,  # loss는 낮을수록 좋음
    logging_steps=50,
    warmup_steps=500,
    fp16=True,
    save_total_limit=2,  # 최대 2개 체크포인트 보관
    gradient_accumulation_steps=2,
    report_to="none",
    lr_scheduler_type="cosine",
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1)[:, 1].numpy()
    return {'roc_auc': roc_auc_score(labels, predictions)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # 검증 데이터셋 추가
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # 조기 종료
)

print(f"monologg/koelectra-base-v3-discriminator 모델 학습 시작...")
trainer.train()

best_model_path = f'{model_output_dir}/best_{config["model_type"]}_soft_cleaning'
model.save_pretrained(best_model_path)
tokenizer.save_pretrained(best_model_path)

models_and_predictions.append({
    'model': model,
    'tokenizer': tokenizer,
    'trainer': trainer,
    'config': config,
    'val_auc': None,
    'save_path': best_model_path
})

```

print(f"\n{'='*50}")
print("monologg/koelectra-base-v3-discriminator 모델 학습 완료")
print(f"{'='*50}")
for model_info in models_and_predictions:
print(f"{model_info['config']['model_type']} 모델 저장 위치: {model_info['save_path']}")
