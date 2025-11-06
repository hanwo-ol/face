import json
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

print(">>> [3단계] 텍스트 토큰화 및 패딩 시작...")

# --- 1. 경로 설정 ---
base_path = r"C:\Users\user\face_recognition\face-1\sub_gentext_siamese"
data_folder = os.path.join(base_path, "gen_text_data")

# 2단계의 산출물을 읽어옵니다.
input_json_file = os.path.join(data_folder, "siamese_data_pairs.json")

# 3단계의 최종 산출물 (다음 4, 5단계에서 사용)
# 1. 토큰화/패딩된 NumPy 데이터
output_npz_file = os.path.join(data_folder, "tokenized_data.npz")
# 2. 토크나이저 자체 (나중에 새로운 텍스트를 판별할 때 필요)
output_tokenizer_file = os.path.join(data_folder, "tokenizer.json")
# 3. 모델 구축 시 필요한 주요 설정값
output_config_file = os.path.join(data_folder, "model_config.json")

print(f"--- 읽어올 파일: {input_json_file}")
print(f"--- 저장할 파일 1 (데이터): {output_npz_file}")
print(f"--- 저장할 파일 2 (토크나이저): {output_tokenizer_file}")
print(f"--- 저장할 파일 3 (설정): {output_config_file}")

# --- 2. 하이퍼파라미터 설정 ---
# 단어 사전에 포함할 최대 단어 개수
MAX_VOCAB_SIZE = 20000 
# 텍스트 시퀀스의 최대 길이 (이보다 길면 자르고, 짧으면 0으로 채움)
MAX_LEN = 150 

# 간단한 텍스트 정제 함수
def clean_text(text):
    text = str(text).lower() # 소문자화
    text = re.sub(r'[^a-z0-9\s]', '', text) # 간단한 특수문자 제거 (영문/숫자/공백만)
    return text

try:
    # --- 3. 2단계에서 저장한 데이터 쌍 로드 ---
    print("\n>>> 2단계의 JSON 데이터 쌍을 로드합니다...")
    with open(input_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pairs = data.get("pairs", []) # [["text_a1", "text_b1"], ["text_a2", "text_b2"], ...]
    labels = data.get("labels", [])   # [0, 1, 0, ...]

    if not pairs or not labels:
        raise ValueError("JSON 파일에서 'pairs' 또는 'labels'를 찾을 수 없습니다. 2단계를 다시 실행하세요.")
    
    print(f"--- 총 {len(pairs)}개의 데이터 쌍 로드 완료.")

    # --- 4. 텍스트 정제 및 Tokenizer 학습 ---
    print("\n>>> 텍스트 정제 및 Tokenizer 학습을 시작합니다...")
    
    # Tokenizer가 학습할 수 있도록 모든 텍스트를 하나의 리스트로 펼칩니다.
    # [text_a1, text_b1, text_a2, text_b2, ...]
    all_texts = [clean_text(text) for pair in pairs for text in pair]

    # Keras Tokenizer 초기화
    # num_words: 가장 빈번한 20000개 단어만 사용
    # oov_token: 20000개에 포함되지 않은 단어는 "<OOV>" (Out-of-Vocabulary)로 처리
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    
    # 단어 사전을 만듭니다.
    tokenizer.fit_on_texts(all_texts)
    
    # 단어 사전 (word -> index 맵)
    word_index = tokenizer.word_index
    
    # 실제 Embedding Layer에 사용할 단어 개수
    # (word_index 개수 + 1 (0번 패딩 토큰용)) 와 MAX_VOCAB_SIZE 중 작은 값
    vocab_size = min(len(word_index) + 1, MAX_VOCAB_SIZE)
    
    print(f"--- Tokenizer 학습 완료. 총 단어 수: {len(word_index)}")
    print(f"--- 모델에 사용할 Vocab Size: {vocab_size}")

    # --- 5. 텍스트를 시퀀스로 변환 및 패딩 ---
    print("\n>>> 텍스트를 정수 시퀀스로 변환하고 패딩을 적용합니다...")
    
    # 텍스트 쌍을 (text_a 리스트)와 (text_b 리스트)로 분리
    texts_a = [clean_text(pair[0]) for pair in pairs]
    texts_b = [clean_text(pair[1]) for pair in pairs]
    
    # Tokenizer를 사용해 텍스트를 정수 시퀀스로 변환
    sequences_a = tokenizer.texts_to_sequences(texts_a)
    sequences_b = tokenizer.texts_to_sequences(texts_b)
    
    # Keras pad_sequences를 사용해 모든 시퀀스의 길이를 MAX_LEN으로 통일
    # padding='post': 0을 뒤쪽에 채움
    # truncating='post': MAX_LEN보다 길 경우 뒤쪽을 자름
    padded_a = pad_sequences(sequences_a, maxlen=MAX_LEN, padding='post', truncating='post')
    padded_b = pad_sequences(sequences_b, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # 두 개의 패딩된 시퀀스를 (N, 2, MAX_LEN) 형태로 합칩니다.
    # axis=1: [padded_a, padded_b] -> [[padded_a[0], padded_b[0]], [padded_a[1], padded_b[1]], ...]
    stacked_pairs = np.stack([padded_a, padded_b], axis=1)
    labels = np.array(labels)
    
    print("--- 시퀀스 변환 및 패딩 완료.")
    print(f"  - Padded 'pairs' 데이터 형태: {stacked_pairs.shape}") # (총 샘플 수, 2, MAX_LEN)
    print(f"  - 'labels' 데이터 형태: {labels.shape}")          # (총 샘플 수,)

    # --- 6. 최종 산출물 저장 ---
    print("\n>>> 최종 데이터를 파일로 저장합니다...")
    
    # 1. 토큰화/패딩된 데이터 (NumPy 압축 포맷)
    np.savez_compressed(
        output_npz_file,
        pairs=stacked_pairs,
        labels=labels
    )
    print(f"  - (성공) NPZ 데이터 저장: {output_npz_file}")
    
    # 2. Tokenizer 저장 (JSON)
    tokenizer_json = tokenizer.to_json()
    with open(output_tokenizer_file, 'w', encoding='utf-8') as f:
        f.write(tokenizer_json)
    print(f"  - (성공) Tokenizer 저장: {output_tokenizer_file}")
        
    # 3. 모델 설정값 저장 (JSON)
    config_data = {
        "vocab_size": vocab_size,
        "max_len": MAX_LEN
    }
    with open(output_config_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4)
    print(f"  - (성공) 모델 설정 저장: {output_config_file}")

    print(f"\n>>> [3단계 성공] 모든 전처리 및 저장 완료.")
    print("    다음 4단계 스크립트에서 이 파일들을 사용합니다.")

except FileNotFoundError:
    print(f"\n!!! 치명적 에러: 2단계의 산출물({input_json_file})을 찾을 수 없습니다.")
    print("    먼저 '02_pair_data_generator.py'를 실행해주세요.")
except ValueError as ve:
    print(f"\n!!! 데이터 값 에러: {ve}")
except Exception as e:
    print(f"\n!!! 3단계 처리 중 예외 발생: {e}")