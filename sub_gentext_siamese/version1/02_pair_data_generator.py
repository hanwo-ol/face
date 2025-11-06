import json
import os
import random
import numpy as np

print(">>> [2단계] 데이터 쌍(Pair) 생성기 시작...")

# --- 1. 경로 설정 ---
# 1단계의 경로와 동일하게 설정합니다.
base_path = r"C:\Users\user\face_recognition\face-1\sub_gentext_siamese"
data_folder = os.path.join(base_path, "gen_text_data")

# 1단계에서 생성된 전처리된 텍스트 파일을 읽어옵니다.
input_json_file = os.path.join(data_folder, "preprocessed_texts.json")

# 2단계의 최종 산출물로, 다음 3단계에서 이 파일을 사용합니다.
output_json_file = os.path.join(data_folder, "siamese_data_pairs.json")

print(f"--- 읽어올 파일: {input_json_file}")
print(f"--- 저장할 파일: {output_json_file}")

try:
    # --- 2. 1단계에서 저장한 텍스트 데이터 로드 ---
    print("\n>>> 1단계에서 전처리된 JSON 파일을 로드합니다...")
    with open(input_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    human_texts = data.get("human_texts", [])
    ai_texts = data.get("ai_texts", [])

    if not human_texts or not ai_texts:
        raise ValueError("'human_texts' 또는 'ai_texts' 리스트가 비어있습니다. 1단계를 다시 실행하세요.")

    print(f"  - 'human' 텍스트 로드 완료: {len(human_texts)}개")
    print(f"  - 'AI' 텍스트 로드 완료: {len(ai_texts)}개")

    # --- 3. 데이터 쌍 생성 (핵심) ---
    print("\n>>> 데이터 쌍 생성을 시작합니다...")

    pairs = []
    labels = []

    # 데이터셋의 균형을 맞추기 위해 더 적은 쪽의 개수를 기준으로 삼습니다.
    min_count = min(len(human_texts), len(ai_texts))
    if min_count == 0:
        raise ValueError("데이터 쌍을 만들기에 텍스트가 부족합니다.")
        
    print(f"--- 각 클래스(긍정/부정)당 {min_count}개의 쌍을 생성합니다.")

    # 1. 부정 쌍 (Negative Pairs, Label = 0) 생성: [Human, AI]
    #    min_count 개 생성
    for i in range(min_count):
        # 각 리스트에서 무작위로 텍스트를 하나씩 선택
        human_text = random.choice(human_texts)
        ai_text = random.choice(ai_texts)
        
        pairs.append([human_text, ai_text])
        labels.append(0)

    # 2. 긍정 쌍 (Positive Pairs, Label = 1) 생성: [Human, Human] + [AI, AI]
    #    총 min_count 개 생성 (각각 절반씩)
    
    # 2-A. [Human, Human] 쌍 생성
    n_pos_human = min_count // 2
    for i in range(n_pos_human):
        # human 리스트에서 2개 무작위 선택 (동일한 것일 수 있음)
        text1 = random.choice(human_texts)
        text2 = random.choice(human_texts)
        pairs.append([text1, text2])
        labels.append(1)

    # 2-B. [AI, AI] 쌍 생성
    # (min_count가 홀수일 경우를 대비해 나머지를 모두 AI 쌍으로 만듦)
    n_pos_ai = min_count - n_pos_human 
    for i in range(n_pos_ai):
        # ai 리스트에서 2개 무작위 선택
        text1 = random.choice(ai_texts)
        text2 = random.choice(ai_texts)
        pairs.append([text1, text2])
        labels.append(1)

    total_pairs = len(pairs)
    print(f"\n>>> 총 {total_pairs}개의 데이터 쌍 생성 완료.")
    print(f"  - 부정 쌍 (Label 0) 개수: {labels.count(0)}")
    print(f"  - 긍정 쌍 (Label 1) 개수: {labels.count(1)}")

    # --- 4. 데이터 셔플 (매우 중요) ---
    # 현재 데이터는 [부정, 부정, ..., 긍정, 긍정] 순서로 정렬되어 있음.
    # 이를 무작위로 섞어야 모델 학습이 제대로 이루어집니다.
    print("\n>>> 생성된 데이터 쌍을 무작위로 섞습니다 (Shuffling)...")
    
    # pairs와 labels를 함께 묶은 뒤 섞어줍니다.
    combined = list(zip(pairs, labels))
    random.shuffle(combined)
    
    # 다시 pairs와 labels로 분리합니다.
    pairs, labels = zip(*combined)
    
    # zip(*...)의 결과는 튜플이므로, 저장을 위해 리스트로 변환합니다.
    pairs = list(pairs)
    labels = list(labels)
    
    print("--- 셔플 완료. (샘플 5개)")
    print(f"  - Pair 1: {pairs[0][0][:30]}... vs {pairs[0][1][:30]}... | Label: {labels[0]}")
    print(f"  - Pair 2: {pairs[1][0][:30]}... vs {pairs[1][1][:30]}... | Label: {labels[1]}")

    # --- 5. 최종 데이터 쌍 저장 ---
    print(f"\n>>> 셔플된 데이터 쌍을 JSON 파일로 저장합니다...")
    
    data_to_save = {
        "pairs": pairs,
        "labels": labels
    }

    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)

    print(f"\n>>> [2단계 성공] 데이터 쌍 저장 완료!")
    print(f"   {output_json_file}")
    print("   다음 3단계 스크립트에서 이 파일을 사용합니다.")

except FileNotFoundError:
    print(f"\n!!! 치명적 에러: 1단계의 산출물({input_json_file})을 찾을 수 없습니다.")
    print("    먼저 '01_load_and_preprocess.py'를 실행해주세요.")
except ValueError as ve:
    print(f"\n!!! 데이터 값 에러: {ve}")
except Exception as e:
    print(f"\n!!! 2단계 처리 중 예외 발생: {e}")