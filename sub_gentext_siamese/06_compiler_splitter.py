import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, RMSprop
import keras.ops as ops
import keras.saving # Keras 사용자 정의 객체 로드를 위해 임포트

print(">>> [6단계] 모델 컴파일 및 데이터 분리 시작...")

# --- 1. ★★★ 실험 하이퍼파라미터 설정 ★★★ ---

# 4/5단계와 동일한 설정을 사용해야 합니다.
EMBEDDING_DIM = 128 
RNN_TYPE = 'LSTM' 
RNN_UNITS = 64
FEATURE_DIM = 32
DISTANCE_METRIC = 'euclidean' 

# --- 이번 단계의 핵심 설정 ---
# Contrastive Loss의 마진 값
# (거리가 이 값보다 멀어지도록 '다른 쌍'을 밀어내는 힘)
MARGIN = 1.0 

# 옵티마이저 선택 ('adam' 또는 'rmsprop')
OPTIMIZER_NAME = 'adam'
LEARNING_RATE = 0.001 # 0.001 (Adam) 또는 0.0001 (RMSprop)이 좋은 시작점

# 데이터 분리 비율
TEST_SPLIT_SIZE = 0.2 # 20%를 테스트 세트로 사용
RANDOM_STATE = 42   # 항상 동일한 방식으로 섞고 분리하기 위함

# --- 2. 경로 설정 ---
base_path = r"C:\Users\user\face_recognition\face-1\sub_gentext_siamese"
data_folder = os.path.join(base_path, "gen_text_data")
model_folder = os.path.join(base_path, "models")

# 3단계에서 저장한 토큰/패딩된 데이터 (입력)
input_data_file = os.path.join(data_folder, "tokenized_data.npz")

# 5단계에서 저장한 샴 네트워크 모델 (입력)
siamese_model_name = f"siamese_model_emb{EMBEDDING_DIM}_{RNN_TYPE.lower()}{RNN_UNITS}_feat{FEATURE_DIM}_{DISTANCE_METRIC}.keras"
model_file = os.path.join(model_folder, siamese_model_name)

# 6단계의 최종 산출물 (7단계에서 사용)
output_split_data_file = os.path.join(data_folder, "train_test_split_data.npz")

print(f"--- 3단계 데이터 (입력): {input_data_file}")
print(f"--- 5단계 모델 (입력/출력): {model_file}")
print(f"--- 6단계 분리된 데이터 (출력): {output_split_data_file}")

# --- 3. Keras 사용자 정의 함수 정의 (모델 로드를 위해 필수) ---
# Keras가 load_model()을 실행할 때 이 함수들을 인식할 수 있도록 데코레이터를 추가합니다.

@keras.saving.register_keras_serializable()
def euclidean_distance(vects):
    """ 두 벡터 사이의 유클리드 거리(L2 Norm) 계산 """
    x, y = vects
    sum_square = ops.sum(ops.square(x - y), axis=1, keepdims=True)
    return ops.sqrt(ops.maximum(sum_square, ops.epsilon()))

@keras.saving.register_keras_serializable()
def manhattan_distance(vects):
    """ 두 벡터 사이의 맨해튼 거리(L1 Norm) 계산 """
    x, y = vects
    return ops.sum(ops.abs(x - y), axis=1, keepdims=True)

@keras.saving.register_keras_serializable()
def cosine_distance(vects):
    """ 
    두 벡터 사이의 코사인 거리 계산 (1 - 코사인 유사도)
    """
    x, y = vects
    x = ops.nn.l2_normalize(x, axis=1)
    y = ops.nn.l2_normalize(y, axis=1)
    similarity = ops.sum(x * y, axis=1, keepdims=True)
    return 1.0 - similarity

@keras.saving.register_keras_serializable()
def get_output_shape(shapes):
    """ Lambda 레이어의 출력 형태를 정의 (항상 (batch_size, 1)) """
    shape1, shape2 = shapes
    return (shape1[0], 1)

@keras.saving.register_keras_serializable()
def contrastive_loss(y_true, y_pred):
    """
    Contrastive Loss 함수.
    y_true (레이블): 1 (같은 쌍), 0 (다른 쌍)
    y_pred (모델 예측): 두 벡터 간의 거리
    """
    # y_true를 y_pred와 동일한 부동소수점 타입으로 변환
    y_true = ops.cast(y_true, y_pred.dtype)
    
    # 1. 같은 쌍 (y_true == 1)의 손실:
    #    거리가 0에 가까워지도록 (distance^2)
    square_pred = ops.square(y_pred)
    
    # 2. 다른 쌍 (y_true == 0)의 손실:
    #    거리가 MARGIN보다 커지도록 (max(MARGIN - distance, 0)^2)
    margin_square = ops.square(ops.maximum(MARGIN - y_pred, 0))
    
    # 3. 두 손실을 y_true 값에 따라 조합
    # y_true=1이면 margin_square가 0이 되고, y_true=0이면 square_pred가 0이 됨
    return ops.mean(y_true * square_pred + (1 - y_true) * margin_square)

# --- 4. 메인 실행 로직 ---
try:
    print(f"\n>>> 3단계 데이터({os.path.basename(input_data_file)}) 로드 중...")
    # 3단계에서 저장한 NPZ 파일 로드
    with np.load(input_data_file) as data:
        pairs = data['pairs']     # (N, 2, MAX_LEN) 형태
        labels = data['labels']   # (N,) 형태
    
    if pairs.shape[0] != labels.shape[0]:
        raise ValueError("데이터 쌍(pairs)과 레이블(labels)의 샘플 수가 일치하지 않습니다.")
    
    print(f"--- 총 {len(labels)}개의 샘플 로드 완료.")

    # --- 5. 데이터 분리 (Train / Test Split) ---
    print(f"\n>>> 데이터를 Train({(1-TEST_SPLIT_SIZE)*100}%) / Test({TEST_SPLIT_SIZE*100}%) 세트로 분리합니다...")
    
    # Scikit-learn의 train_test_split 사용
    # stratify=labels : 원본 데이터의 0/1 비율을 Train/Test 세트에도 동일하게 유지
    pairs_train, pairs_test, labels_train, labels_test = train_test_split(
        pairs, labels,
        test_size=TEST_SPLIT_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels 
    )
    
    # 모델의 두 입력(input_a, input_b)에 맞게 데이터 분리
    # pairs_train (N, 2, MAX_LEN) -> X_train_a (N, MAX_LEN), X_train_b (N, MAX_LEN)
    X_train_a = pairs_train[:, 0]
    X_train_b = pairs_train[:, 1]
    y_train = labels_train
    
    X_test_a = pairs_test[:, 0]
    X_test_b = pairs_test[:, 1]
    y_test = labels_test
    
    print("--- 데이터 분리 완료.")
    print(f"  - 학습(Train) 세트: {len(y_train)}개")
    print(f"  - 테스트(Test) 세트: {len(y_test)}개")

    # --- 6. 분리된 데이터 저장 (7단계용) ---
    print(f"\n>>> 분리된 데이터를 '{os.path.basename(output_split_data_file)}' 파일로 저장합니다...")
    np.savez_compressed(
        output_split_data_file,
        X_train_a=X_train_a,
        X_train_b=X_train_b,
        y_train=y_train,
        X_test_a=X_test_a,
        X_test_b=X_test_b,
        y_test=y_test
    )
    print(f"--- 데이터 저장 완료.")

    # --- 7. 모델 로드, 옵티마이저 선택 및 컴파일 ---
    print(f"\n>>> 5단계 모델({os.path.basename(model_file)})을 로드합니다...")
    
    # ★★★ 중요 ★★★
    # load_model을 호출하기 전에 위(3단계)에서 사용자 정의 함수들이
    # @register_keras_serializable과 함께 정의되어 있어야 합니다.
    model = load_model(model_file)
    
    print("--- 모델 로드 완료.")

    # 옵티마이저 선택
    print(f"--- 옵티마이저 설정: {OPTIMIZER_NAME} (Learning Rate={LEARNING_RATE})")
    if OPTIMIZER_NAME.lower() == 'adam':
        optimizer = Adam(learning_rate=LEARNING_RATE)
    elif OPTIMIZER_NAME.lower() == 'rmsprop':
        optimizer = RMSprop(learning_rate=LEARNING_RATE)
    else:
        print(f"경고: {OPTIMIZER_NAME}을(를) 찾을 수 없습니다. 기본 'adam'을 사용합니다.")
        optimizer = Adam(learning_rate=LEARNING_RATE)

    # 모델 컴파일
    print(f"\n>>> 'contrastive_loss' (margin={MARGIN})와 옵티마이저로 모델을 컴파일합니다...")
    model.compile(loss=contrastive_loss, optimizer=optimizer)
    print("--- 모델 컴파일 완료.")
    
    # --- 8. 컴파일된 모델 저장 ---
    print(f"\n>>> 컴파일된 모델을 '{os.path.basename(model_file)}' 파일에 덮어씁니다...")
    # 7단계에서 바로 로드할 수 있도록 컴파일된 상태로 덮어쓰기
    model.save(model_file)
    print("--- 컴파일된 모델 저장 완료.")

    print(f"\n>>> [6단계 성공] 모든 준비 완료.")
    print("   다음 7단계 스크립트에서 학습을 시작할 수 있습니다.")

except FileNotFoundError as e:
    print(f"\n!!! 치명적 에러: 파일({e.filename})을 찾을 수 없습니다.")
    print("    '03_tokenize_data.py' 또는 '05_siamese_model.py'를 먼저 실행했는지 확인하세요.")
except ValueError as ve:
    print(f"\n!!! 데이터 값 에러: {ve}")
except Exception as e:
    print(f"\n!!! 6단계 처리 중 예외 발생: {e}")