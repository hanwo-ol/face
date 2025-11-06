import os
import json
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model, load_model
import keras.ops as ops

print(">>> [5단계] 샴 네트워크(Siamese Network) 모델 조립 시작...")

# --- 1. ★★★ 실험 하이퍼파라미터 설정 ★★★ ---

# 4단계에서 생성한 Base Network와 동일한 설정을 사용해야 합니다.
# 이 값을 변경했다면, 4단계를 다시 실행하여 해당 모델을 생성해야 합니다.
EMBEDDING_DIM = 128 
RNN_TYPE = 'LSTM' 
RNN_UNITS = 64
FEATURE_DIM = 32

# ★★★ 이번 단계의 핵심 선택: 사용할 거리 함수 ★★★
# 'euclidean', 'manhattan', 'cosine' 중 하나를 선택하세요.
DISTANCE_METRIC = 'euclidean' 

# --- 2. 경로 설정 ---
base_path = r"C:\Users\user\face_recognition\face-1\sub_gentext_siamese"
data_folder = os.path.join(base_path, "gen_text_data")
model_folder = os.path.join(base_path, "models")

# 3단계에서 저장한 모델 설정 파일 (max_len을 가져오기 위함)
input_config_file = os.path.join(data_folder, "model_config.json")

# 4단계에서 생성된, 로드할 Base Network 파일 이름
base_model_name = f"base_network_emb{EMBEDDING_DIM}_{RNN_TYPE.lower()}{RNN_UNITS}_feat{FEATURE_DIM}.keras"
input_base_model_file = os.path.join(model_folder, base_model_name)

# 5단계의 최종 산출물 (전체 샴 네트워크 모델)
siamese_model_name = f"siamese_model_emb{EMBEDDING_DIM}_{RNN_TYPE.lower()}{RNN_UNITS}_feat{FEATURE_DIM}_{DISTANCE_METRIC}.keras"
output_siamese_model_file = os.path.join(model_folder, siamese_model_name)

print(f"--- 3단계 설정 파일: {input_config_file}")
print(f"--- 4단계 Base Network (입력): {input_base_model_file}")
print(f"--- 5단계 Siamese Model (출력): {output_siamese_model_file}")

# --- 3. 거리 함수 및 출력 형태 정의 ---

def euclidean_distance(vects):
    """ 두 벡터 사이의 유클리드 거리(L2 Norm) 계산 """
    x, y = vects
    # ops.epsilon() : 0으로 나누거나 0의 제곱근을 구하는 것을 방지하기 위한 아주 작은 값
    sum_square = ops.sum(ops.square(x - y), axis=1, keepdims=True)
    return ops.sqrt(ops.maximum(sum_square, ops.epsilon()))

def manhattan_distance(vects):
    """ 두 벡터 사이의 맨해튼 거리(L1 Norm) 계산 """
    x, y = vects
    return ops.sum(ops.abs(x - y), axis=1, keepdims=True)

def cosine_distance(vects):
    """ 
    두 벡터 사이의 코사인 거리 계산 (1 - 코사인 유사도)
    유사도(Similarity)는 1에 가까울수록 비슷, 거리(Distance)는 0에 가까울수록 비슷.
    """
    x, y = vects
    
    # 각 벡터를 L2 정규화 (크기를 1로 만듦)
    x = ops.nn.l2_normalize(x, axis=1)
    y = ops.nn.l2_normalize(y, axis=1)
    
    # 코사인 유사도 계산 (벡터 내적)
    similarity = ops.sum(x * y, axis=1, keepdims=True)
    
    # 코사인 거리 (1 - 유사도) 반환
    return 1.0 - similarity

def get_output_shape(shapes):
    """ Lambda 레이어의 출력 형태를 정의 (항상 (batch_size, 1)) """
    shape1, shape2 = shapes
    return (shape1[0], 1)

# --- 4. 메인 실행 로직 ---
try:
    print(f"\n>>> 3단계 설정 파일({os.path.basename(input_config_file)}) 로드 중...")
    with open(input_config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    max_len = config.get("max_len")
    if not max_len:
        raise ValueError("설정 파일에 'max_len'이 없습니다.")
    print(f"--- 'max_len' 로드 완료: {max_len}")
    
    
    print(f"\n>>> 4단계 Base Network({base_model_name}) 로드 중...")
    base_network = load_model(input_base_model_file)
    # base_network.summary() # (옵션) 로드된 모델 구조 확인
    print("--- Base Network 로드 완료.")

    # --- 5. 샴 네트워크 조립 ---
    print("\n>>> 전체 샴 네트워크 모델을 조립합니다...")

    # 1. 두 개의 입력층을 정의 (얼굴 인식 실습과 동일)
    # 입력 형태는 (max_len,)
    input_a = Input(shape=(max_len,), name="input_text_a")
    input_b = Input(shape=(max_len,), name="input_text_b")
    
    # 2. 두 입력이 *동일한* Base Network를 통과 (가중치 공유)
    feat_vecs_a = base_network(input_a)
    feat_vecs_b = base_network(input_b)
    
    # 3. 설정에 따라 거리 함수 선택
    print(f"--- 적용할 거리 함수: {DISTANCE_METRIC}")
    if DISTANCE_METRIC == 'euclidean':
        selected_distance_func = euclidean_distance
    elif DISTANCE_METRIC == 'manhattan':
        selected_distance_func = manhattan_distance
    elif DISTANCE_METRIC == 'cosine':
        selected_distance_func = cosine_distance
    else:
        raise ValueError(f"지원하지 않는 거리 함수입니다: {DISTANCE_METRIC}")

    # 4. Lambda 레이어를 사용해 두 특징 벡터의 거리를 계산
    distance = Lambda(
        selected_distance_func,
        output_shape=get_output_shape,
        name="distance_lambda"
    )([feat_vecs_a, feat_vecs_b])

    # 5. 최종 샴 네트워크 모델 생성
    # (얼굴 인식 실습의 ValueError를 반영하여 inputs/outputs 사용)
    model = Model(
        inputs=[input_a, input_b], 
        outputs=distance,
        name="Siamese_Text_Model"
    )

    # --- 6. 모델 요약 및 저장 ---
    print("\n>>> 최종 조립된 샴 네트워크 모델 구조:")
    model.summary()

    print(f"\n>>> 샴 네트워크 모델을 파일로 저장합니다...")
    model.save(output_siamese_model_file)

    print(f"\n>>> [5단계 성공] 샴 네트워크 모델 저장 완료!")
    print(f"   {output_siamese_model_file}")
    print("   다음 6단계 스크립트에서 이 모델을 로드하여 학습을 진행합니다.")

except FileNotFoundError:
    print(f"\n!!! 치명적 에러: 3단계({input_config_file}) 또는 4단계({input_base_model_file})의 산출물을 찾을 수 없습니다.")
    print("    먼저 '03_tokenize_data.py'와 '04_make_base_network.py'를 실행해주세요.")
except ValueError as ve:
    print(f"\n!!! 데이터 값 에러: {ve}")
except Exception as e:
    print(f"\n!!! 5단계 처리 중 예외 발생: {e}")