import json
from gensim.models import KeyedVectors
import numpy as np
import random


# loading word list
with open("korean_frequency.json", encoding="utf-8") as f:
    data = json.load(f)

# list concatenating
word_list = sorted({w for group in data.values() for w in group})

# loading FastText model
model = KeyedVectors.load("cc.ko.300.kv")

# updating candidates


def update_candidates(candidates, guess, similarity, tolerance=5.0):
    new_candidates = []
    for cand in candidates:
        if cand not in model or guess not in model:
            continue
        sim = model.similarity(cand, guess) * 100
        if abs(sim - similarity) <= tolerance:
            new_candidates.append(cand)
    return new_candidates

# selecting best


def select_best_guess(candidates, tried):
    max_entropy = -1
    best_word = None
    for word in candidates:
        if word not in model:
            continue
        sims = []
        for other in tried:
            if other in model:
                sims.append(model.similarity(word, other))
        entropy = -np.sum(np.log(np.clip(sims, 1e-5, 1))) if sims else 0
        if entropy > max_entropy:
            max_entropy = entropy
            best_word = word
    return best_word or random.choice(candidates)


def simulate_game(answer):
    candidates = [w for w in word_list if w in model]
    tried = []
    for step in range(1, 21):
        guess = select_best_guess(candidates, tried)
        tried.append(guess)

        if guess == answer:
            print(f"✅ 정답 '{guess}'을 {step}번 만에 맞췄습니다!")
            return tried

        similarity = model.similarity(guess, answer) * 100
        print(f"{step}. '{guess}' → 유사도: {similarity:.2f}")
        candidates = update_candidates(candidates, guess, similarity)

    print("❌ 실패: 20회 이내에 정답을 찾지 못했습니다.")
    return tried


def start_interactive_guessing():
    candidates = [w for w in word_list if w in model]
    tried = []

    print("🎯 꼬맨틀 정답 예측 도우미 시작!")
    print("유사도는 정수 또는 소수점(예: 74.3)으로 입력하세요.")
    print("정답을 맞혔다면 '정답'이라고 입력해 주세요.\n")

    for step in range(1, 21):
        guess = select_best_guess(candidates, tried)
        print(f"{step}. 추천 추측 단어: '{guess}'")

        user_input = input("→ 정답과의 유사도 점수는? (또는 '정답') : ").strip()

        if user_input.lower() in ['정답', '맞춤', '끝']:
            print(f"\n🎉 정답: '{guess}' (총 {step}회 시도)")
            return tried + [guess]

        try:
            similarity = float(user_input)
        except ValueError:
            print("⚠️ 숫자 또는 '정답'만 입력해 주세요.")
            continue

        tried.append(guess)
        candidates = update_candidates(candidates, guess, similarity)
        print(f"남은 후보 수: {len(candidates)}개\n")

        if len(candidates) == 0:
            print("❌ 후보가 없습니다. 유사도 입력을 확인해 주세요.")
            break

    print("🔚 20회 이상 시도됨. 정답 유추 실패.")
    return tried


start_interactive_guessing()
