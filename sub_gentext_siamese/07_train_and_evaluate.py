import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.ops as ops
import keras.saving # Keras 사용자 정의 객체 로드를 위해 임포트
import matplotlib.pyplot as plt

print(">>> [7단계] 모델 학습 및 평가 시작...")

# --- 1. ★★★ 실험 하이퍼파라미터 설정 ★★★ ---
# 4, 5, 6단계와 동일한 설정을 사용해야 합니다.
EMBEDDING_DIM = 128 
RNN_TYPE = 'LSTM' 
RNN_UNITS = 64
FEATURE_DIM = 32
DISTANCE_METRIC = 'euclidean' 
MARGIN = 1.0 # 6단계의 contrastive_loss 마진과 동일해야 함

# --- 이번 단계의 핵심 설정 ---
EPOCHS = 50        # 최대 학습 횟수
BATCH_SIZE = 32    # 한 번에 학습할 데이터 쌍의 수
PATIENCE = 5       # EarlyStopping: 5번 연속으로 val_loss가 향상되지 않으면 학습 중단
THRESHOLD = 0.5    # 정확도 계산 임계값 (거리가 0.5 미만이면 '같은 쌍'으로 간주)

# --- 2. 경로 설정 ---
base_path = r"C:\Users\user\face_recognition\face-1\sub_gentext_siamese"
data_folder = os.path.join(base_path, "gen_text_data")
model_folder = os.path.join(base_path, "models")
results_folder = os.path.join(base_path, "results")

# 6단계에서 컴파일된 모델 (입력)
siamese_model_name = f"siamese_model_emb{EMBEDDING_DIM}_{RNN_TYPE.lower()}{RNN_UNITS}_feat{FEATURE_DIM}_{DISTANCE_METRIC}.keras"
compiled_model_file = os.path.join(model_folder, siamese_model_name)

# 6단계에서 분리된 데이터 (입력)
split_data_file = os.path.join(data_folder, "train_test_split_data.npz")

# 7단계의 최종 산출물 (학습된 최적의 모델)
trained_model_name = f"trained_best_model_emb{EMBEDDING_DIM}_{RNN_TYPE.lower()}{RNN_UNITS}_feat{FEATURE_DIM}_{DISTANCE_METRIC}.keras"
best_model_file = os.path.join(model_folder, trained_model_name)

# 7단계의 최종 산출물 (학습 그래프)
plot_name = f"training_history_emb{EMBEDDING_DIM}_{RNN_TYPE.lower()}{RNN_UNITS}_feat{FEATURE_DIM}_{DISTANCE_METRIC}.png"
plot_file = os.path.join(results_folder, plot_name)

# 결과 폴더가 없으면 생성
os.makedirs(results_folder, exist_ok=True)

print(f"--- 6단계 데이터 (입력): {split_data_file}")
print(f"--- 6단계 모델 (입력): {compiled_model_file}")
print(f"--- 7단계 최적 모델 (출력): {best_model_file}")
print(f"--- 7단계 학습 그래프 (출력): {plot_file}")


# --- 3. Keras 사용자 정의 함수 정의 (모델 로드를 위해 필수) ---
# (6단계에서 복사)
@keras.saving.register_keras_serializable()
def euclidean_distance(vects):
    x, y = vects
    sum_square = ops.sum(ops.square(x - y), axis=1, keepdims=True)
    # --- 수정된 부분 ---
    # ops.epsilon() 대신 Keras의 기본 epsilon 값인 1e-7을 직접 사용합니다.
    return ops.sqrt(ops.maximum(sum_square, 1e-7))

@keras.saving.register_keras_serializable()
def manhattan_distance(vects):
    x, y = vects
    return ops.sum(ops.abs(x - y), axis=1, keepdims=True)

@keras.saving.register_keras_serializable()
def cosine_distance(vects):
    x, y = vects
    x = ops.nn.l2_normalize(x, axis=1)
    y = ops.nn.l2_normalize(y, axis=1)
    similarity = ops.sum(x * y, axis=1, keepdims=True)
    return 1.0 - similarity

@keras.saving.register_keras_serializable()
def get_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

@keras.saving.register_keras_serializable()
def contrastive_loss(y_true, y_pred):
    y_true = ops.cast(y_true, y_pred.dtype)
    square_pred = ops.square(y_pred)
    margin_square = ops.square(ops.maximum(MARGIN - y_pred, 0))
    return ops.mean(y_true * square_pred + (1 - y_true) * margin_square)

# --- 4. 정확도 계산 함수 정의 ---
# (얼굴 인식 실습에서 가져온 함수)
def compute_accuracy(predictions, labels, threshold=0.5):
    """
    임계값(threshold)을 기준으로 정확도를 계산합니다.
    (예측 거리 < 임계값)일 때 '같은 쌍'(1)이라고 예측한 것으로 간주합니다.
    """
    # 예측 거리가 임계값보다 작으면 1 (같은 쌍), 크면 0 (다른 쌍)으로 이진화
    preds_binary = (predictions.ravel() < threshold).astype(int)
    
    # 이진화된 예측과 실제 레이블(labels)이 얼마나 일치하는지 계산
    accuracy = np.mean(preds_binary == labels)
    return accuracy

# --- 5. 메인 학습 및 평가 로직 ---
try:
    # --- 5-1. 데이터 로드 ---
    print(f"\n>>> 6단계 데이터({os.path.basename(split_data_file)}) 로드 중...")
    with np.load(split_data_file) as data:
        X_train_a = data['X_train_a']
        X_train_b = data['X_train_b']
        y_train = data['y_train']
        X_test_a = data['X_test_a']
        X_test_b = data['X_test_b']
        y_test = data['y_test']
    
    print(f"--- 학습 세트 {len(y_train)}개, 테스트 세트 {len(y_test)}개 로드 완료.")

    # --- 5-2. 모델 로드 ---
    print(f"\n>>> 6단계 컴파일된 모델({os.path.basename(compiled_model_file)}) 로드 중...")
    model = load_model(compiled_model_file)
    print("--- 모델 로드 완료.")
    model.summary() # 로드된 모델 구조 확인

    # --- 5-3. 콜백(Callbacks) 정의 ---
    # ModelCheckpoint: 검증 손실(val_loss)이 가장 낮은 모델만 'best_model_file'에 저장
    checkpoint = ModelCheckpoint(
        filepath=best_model_file,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # EarlyStopping: 검증 손실(val_loss)이 PATIENCE 횟수만큼 향상되지 않으면 학습 조기 중단
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        verbose=1
    )
    
    print("\n>>> [ Keras 모델 학습 시작 ] ---")
    print(f"--- Epochs={EPOCHS}, Batch Size={BATCH_SIZE}, Patience={PATIENCE} ---")

    # --- 5-4. 모델 학습 ---
    history = model.fit(
        [X_train_a, X_train_b],  # 학습 데이터 입력 (쌍)
        y_train,                 # 학습 데이터 레이블 (0 또는 1)
        validation_data=([X_test_a, X_test_b], y_test), # 검증 데이터
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early_stopping], # 콜백 적용
        verbose=1
    )

    print(">>> [ 모델 학습 완료 ] ---")

    # --- 5-5. 학습 결과 시각화 ---
    print(f"\n>>> 학습 과정을 '{os.path.basename(plot_file)}' 파일로 저장합니다...")
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Contrastive Loss)')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_file)
    print(f"--- 학습 그래프 저장 완료. (확인: {plot_file})")
    # plt.show() # (주석 처리. 스크립트 환경에서는 show()가 불필요할 수 있음)

    # --- 5-6. 최종 평가 (최적 모델 사용) ---
    print(f"\n>>> [ 최종 평가 시작 ] ---")
    print(f"--- 저장된 최적의 모델({os.path.basename(best_model_file)})을 로드합니다...")
    
    # ModelCheckpoint가 저장한 최적의 모델을 로드
    best_model = load_model(best_model_file)
    
    print(f"\n>>> 테스트 세트(총 {len(y_test)}개)로 예측을 수행합니다...")
    # 테스트 세트로 예측 (거리 계산)
    predictions = best_model.predict([X_test_a, X_test_b])

    # 정확도 계산
    accuracy = compute_accuracy(predictions, y_test, threshold=THRESHOLD)
    
    print("\n--- [ 최종 평가 결과 ] ---")
    print(f"  - 정확도 계산 임계값 (Threshold): {THRESHOLD}")
    print(f"  - 테스트 세트 정확도 (Accuracy): {accuracy * 100:.2f} %")
    print("--------------------------")

    print(f"\n>>> [7단계 성공] 모든 학습 및 평가가 완료되었습니다.")

except FileNotFoundError as e:
    print(f"\n!!! 치명적 에러: 파일({e.filename})을 찾을 수 없습니다.")
    print("    '06_compiler_splitter.py'를 먼저 실행했는지 확인하세요.")
except Exception as e:
    print(f"\n!!! 7단계 처리 중 예외 발생: {e}")