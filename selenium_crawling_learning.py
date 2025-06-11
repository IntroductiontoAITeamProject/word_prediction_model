from gensim.models import Word2Vec
from konlpy.tag import Okt
import time
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
import json

with open("korean_frequency.json", encoding="utf-8") as f:
    data = json.load(f)
target_words = sorted({w for group in data.values() for w in group})


def fetch_sentences_selenium(word, max_pages=1):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # 창 띄우지 않음
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(
        ChromeDriverManager().install()), options=options)
    results = []

    for page in range(1, max_pages + 1):
        start = (page - 1) * 10 + 1
        url = f"https://search.naver.com/search.naver?where=news&query={word}&start={start}"
        print(f"[{word}] 페이지 {page} 크롤링 중...")

        try:
            driver.get(url)

            # 해당 클래스의 텍스트가 뉴스 제목으로 보인다면 이 방식 사용
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, ".sds-comps-text-type-headline1"))
            )
            articles = driver.find_elements(
                By.CSS_SELECTOR, ".sds-comps-text-type-headline1")
            for a in articles:
                text = a.text.strip()
                if text:
                    results.append(text)

        except Exception as e:
            print(f"[{word}] 페이지 {page} 오류: {e}")

    driver.quit()
    print(f"[{word}] 수집 완료 (총 {len(results)}문장)")
    return results


okt = Okt()


def preprocess_sentences(sentences):
    processed = []
    for s in sentences:
        words = okt.morphs(s, stem=True)
        if words:
            processed.append(words)
    return processed


all_sentences = []
for word in target_words[:100]:  # 테스트용: 상위 100개 단어만
    try:
        sents = fetch_sentences_selenium(word, max_pages=2)
        all_sentences.extend(sents)
    except Exception as e:
        print(f"[오류] {word}: {e}")

print(f"총 수집된 문장 수: {len(all_sentences)}")
print("예시 문장:", all_sentences[0] if all_sentences else "없음")


tokenized = preprocess_sentences(all_sentences)

model = Word2Vec(
    sentences=tokenized,
    vector_size=300,
    window=5,
    min_count=1,
    workers=4
)
model.wv.save("wiktionary5800_custom.kv")
print("Word2Vec 저장 완료")
