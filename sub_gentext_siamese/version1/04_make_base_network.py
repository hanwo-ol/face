import os
import json
from tensorflow.keras.layers import Input, Embedding, LSTM, GRU, Dense
from tensorflow.keras.models import Model, Sequential

print(">>> [4단계] 텍스트용 Base Network 구축 시작...")

# --- 1. ★★★ 실험 하이퍼파라미터 설정 ★★★ ---
# 이 값을 변경하면서 4, 5, 6, 7단계를 반복 실행하여 성능을 비교할 수 있습니다.

# 1. 임베딩 벡터의 차원
EMBEDDING_DIM = 128 

# 2. RNN 레이어의 타입 ('LSTM' 또는 'GRU')
RNN_TYPE = 'LSTM' 

# 3. RNN 레이어의 유닛(뉴런) 개수
RNN_UNITS = 64

# 4. 최종 출력될 특징 벡터(Feature Vector)의 차원
FEATURE_DIM = 32

# --- 2. 경로 설정 ---
base_path = r"C:\Users\user\face_recognition\face-1\sub_gentext_siamese"
data_folder = os.path.join(base_path, "gen_text_data")
model_folder = os.path.join(base_path, "models") # 모델을 저장할 폴더

# 3단계에서 저장한 모델 설정 파일을 읽어옵니다.
input_config_file = os.path.join(data_folder, "model_config.json")

# 하이퍼파라미터에 따라 동적으로 모델 파일 이름 생성
# 예: base_network_emb128_lstm64_feat32.keras
model_name = f"base_network_emb{EMBEDDING_DIM}_{RNN_TYPE.lower()}{RNN_UNITS}_feat{FEATURE_DIM}.keras"
output_model_file = os.path.join(model_folder, model_name)

# 모델 저장 폴더가 없으면 생성
os.makedirs(model_folder, exist_ok=True)

print(f"--- 3단계 설정 파일: {input_config_file}")
print(f"--- 생성될 모델 파일: {output_model_file}")


def build_base_network_text(vocab_size, max_len, embedding_dim, rnn_type, rnn_units, feature_dim):
    """
    텍스트 시퀀스를 입력받아 특징 벡터(Feature Vector)를
    출력하는 Base Network를 생성합니다. (Sequential API 사용)
    """
    
    # 1. 입력층 (Input)
    # (max_len,) 형태의 정수 시퀀스를 받습니다.
    text_input = Input(shape=(max_len,), name="text_input")
    
    # 2. 임베딩 레이어 (Embedding)
    # (None, max_len) -> (None, max_len, embedding_dim)
    # 이 레이어가 단어의 '문체'를 학습하게 됩니다.
    embedded_text = Embedding(
        input_dim=vocab_size,       # 3단계에서 계산한 단어 사전 크기
        output_dim=embedding_dim,   # 실험 하이퍼파라미터
        input_length=max_len,
        name="embedding_layer"
    )(text_input)
    
    # 3. RNN 레이어 (LSTM/GRU)
    # (None, max_len, embedding_dim) -> (None, rnn_units)
    # 시퀀스의 문맥과 순서를 압축하여 하나의 벡터로 만듭니다.
    if rnn_type.upper() == 'LSTM':
        rnn_layer = LSTM(rnn_units, name="lstm_layer")(embedded_text)
    elif rnn_type.upper() == 'GRU':
        rnn_layer = GRU(rnn_units, name="gru_layer")(embedded_text)
    else:
        raise ValueError(f"지원하지 않는 RNN 타입입니다: {rnn_type}. 'LSTM' 또는 'GRU'를 사용하세요.")
        
    # 4. 최종 특징 벡터 (Dense)
    # (None, rnn_units) -> (None, feature_dim)
    # 얼굴 인식 실습의 50차원 벡터와 동일한 역할을 합니다.
    feature_vector = Dense(
        feature_dim, 
        activation='relu',  # 비선형성을 추가하여 특징을 더 잘 구분
        name="feature_vector"
    )(rnn_layer)
    
    # Functional API를 사용하여 모델을 생성
    base_network = Model(inputs=text_input, outputs=feature_vector, name="Text_Base_Network")
    
    return base_network

# --- 3. 메인 실행 로직 ---
try:
    print(f"\n>>> 3단계에서 저장한 '{os.path.basename(input_config_file)}' 파일을 로드합니다...")
    
    # 3단계에서 저장한 vocab_size와 max_len을 불러옵니다.
    with open(input_config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    vocab_size = config.get("vocab_size")
    max_len = config.get("max_len")
    
    if not vocab_size or not max_len:
        raise ValueError("설정 파일에 'vocab_size' 또는 'max_len'이 없습니다.")

    print("--- 3단계 설정 로드 완료.")
    print(f"  - Vocab Size (단어 사전 크기): {vocab_size}")
    print(f"  - Max Length (시퀀스 최대 길이): {max_len}")
    
    print("\n>>> 현재 하이퍼파라미터로 Base Network를 구축합니다...")
    print(f"  - EMBEDDING_DIM: {EMBEDDING_DIM}")
    print(f"  - RNN_TYPE: {RNN_TYPE}")
    print(f"  - RNN_UNITS: {RNN_UNITS}")
    print(f"  - FEATURE_DIM: {FEATURE_DIM}")
    
    # 정의된 함수를 호출하여 모델 생성
    base_network = build_base_network_text(
        vocab_size=vocab_size,
        max_len=max_len,
        embedding_dim=EMBEDDING_DIM,
        rnn_type=RNN_TYPE,
        rnn_units=RNN_UNITS,
        feature_dim=FEATURE_DIM
    )
    
    # 구축된 모델의 구조를 요약해서 보여줍니다.
    print("\n>>> Base Network 구조 요약:")
    base_network.summary()
    
    # --- 4. 모델 저장 ---
    print(f"\n>>> 구축된 Base Network를 파일로 저장합니다...")
    base_network.save(output_model_file)
    
    print(f"\n>>> [4단계 성공] Base Network 모델 저장 완료!")
    print(f"   {output_model_file}")
    print("   다음 5단계 스크립트에서 이 모델을 로드하여 샴 네트워크를 조립합니다.")

except FileNotFoundError:
    print(f"\n!!! 치명적 에러: 3단계의 산출물({input_config_file})을 찾을 수 없습니다.")
    print("    먼저 '03_tokenize_data.py'를 실행해주세요.")
except ValueError as ve:
    print(f"\n!!! 데이터 값 에러: {ve}")
except Exception as e:
    print(f"\n!!! 4단계 처리 중 예외 발생: {e}")