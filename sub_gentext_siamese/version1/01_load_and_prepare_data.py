import pandas as pd
import numpy as np
import re
import os
import json  # <- 텍스트 리스트를 저장하기 위해 json 라이브러리 추가

# --- 텍스트 처리를 위한 Keras(TensorFlow) 라이브러리 ---
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 모델 구축을 위한 Keras(TensorFlow) 라이브러리 ---
from tensorflow.keras.layers import Input, Embedding, LSTM, GRU, Dense, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop

# --- 수학 연산 및 손실 함수를 위한 Keras 3.x 라이브러리 ---
import keras.ops as ops

# --- 데이터 분리를 위한 Scikit-learn 라이브러리 ---
from sklearn.model_selection import train_test_split

print(">>> [1단계] 모든 라이브러리 임포트 완료.")

# --- 1. 경로 설정 ---
# 이 스크립트의 목적: 원본 Excel -> 전처리된 JSON
base_path = r"C:\Users\user\face_recognition\face-1\sub_gentext_siamese"
data_folder = os.path.join(base_path, "gen_text_data")

# 원본 데이터 파일 경로
input_excel_file = os.path.join(data_folder, "data_combined.xlsx")

# 1차 전처리된 데이터를 저장할 파일 경로
# 다음 2단계부터는 이 JSON 파일을 읽어 사용합니다.
output_json_file = os.path.join(data_folder, "preprocessed_texts.json")

print(f"--- 원본 데이터 경로: {input_excel_file}")
print(f"--- 저장될 파일 경로: {output_json_file}")

try:
    # --- 2. 원본 Excel 데이터 로드 ---
    # .xlsx 파일이므로 pd.read_excel을 사용합니다.
    print("\n>>> 원본 Excel 파일 로드를 시도합니다... (openpyxl 필요)")
    df = pd.read_excel(input_excel_file, engine='openpyxl')

    print(">>> 데이터 로드 성공. 상위 5개 행:")
    print(df.head())

    # --- 3. 텍스트 데이터 전처리 및 분리 ---
    print("\n>>> 텍스트 전처리 및 분리를 시작합니다...")
    
    # 'Answer' 또는 'model' 열이 없는 경우 오류 발생
    if 'Answer' not in df.columns or 'model' not in df.columns:
        raise ValueError("파일에 'Answer' 또는 'model' 열이 존재하지 않습니다.")

    # 결측치(NaN) 제거
    df = df.dropna(subset=['Answer', 'model'])
    
    # 텍스트와 레이블을 표준화 (str 타입 변환, 양 끝 공백 제거)
    df['Answer'] = df['Answer'].astype(str).str.strip()
    df['model'] = df['model'].astype(str).str.strip()

    # 'human'을 제외한 모든 모델을 'AI'로 간주하여 리스트업
    all_models = df['model'].unique()
    human_texts = df[df['model'] == 'human']['Answer'].tolist()
    
    # 'human'이 아니고, 비어있지 않은 모든 모델 이름을 AI로 분류
    ai_model_names = [m for m in all_models if m and m != 'human']
    
    if not ai_model_names:
        raise ValueError("'human' 이외의 다른 모델 레이블(예: 'claude')을 찾을 수 없습니다.")

    print(f"--- 'AI'로 간주될 모델: {ai_model_names}")
    ai_texts = df[df['model'].isin(ai_model_names)]['Answer'].tolist()

    print("\n>>> 텍스트 분리 완료:")
    print(f"  - 총 'human' 텍스트 개수: {len(human_texts)}")
    print(f"  - 총 'AI' 텍스트 개수: {len(ai_texts)}")

    # --- 4. 전처리된 데이터 저장 (핵심) ---
    print(f"\n>>> 전처리된 텍스트 리스트를 JSON 파일로 저장합니다...")
    
    data_to_save = {
        "human_texts": human_texts,
        "ai_texts": ai_texts
    }

    # utf-8 인코딩으로 JSON 파일 저장
    with open(output_json_file, 'w', encoding='utf-8') as f:
        # ensure_ascii=False: 한글 등이 깨지지 않고 저장되도록 함
        # indent=4: 파일을 열었을 때 보기 편하도록 들여쓰기 적용
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)

    print(f"\n>>> [1단계 성공] 데이터 저장 완료!")
    print(f"   {output_json_file}")
    print("   다음 2단계 스크립트에서 이 파일을 사용합니다.")

except FileNotFoundError:
    print(f"\n!!! 치명적 에러: 원본 파일을 찾을 수 없습니다.")
    print(f"    경로를 다시 확인해주세요: {input_excel_file}")
except ImportError:
    print("\n!!! 치명적 에러: 'openpyxl' 라이브러리가 필요합니다.")
    print("    터미널에서 'pip install openpyxl'을 실행해주세요.")
except ValueError as ve:
    print(f"\n!!! 데이터 값 에러: {ve}")
except Exception as e:
    print(f"\n!!! 1단계 처리 중 예외 발생: {e}")