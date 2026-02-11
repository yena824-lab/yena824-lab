import pandas as pd
from tqdm import tqdm
from transformers import pipeline
import os
import random
import numpy as np
import gc
import torch

# === 데이터 불러오기 ===
csv_data = pd.read_csv("/home/yena/Classification_module/val_data.csv", encoding="utf-8-sig")
csv_data["제목+내용"] = csv_data["제목"] + ' ' + csv_data["내용"]
total_data = csv_data[["제목+내용", "대분류", "중분류"]]
classification_data = pd.read_excel("/home/yena/food_classification/식품유형 및 원인요소 분류표.xlsx", sheet_name='원인요소')

# === 환경 설정 ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 사용할 GPU 지정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 메모리 조각화 방지

# === 모델 로딩 (Hugging Face Llama 3-3B Instruct) ===
model_id = "/home/yena/Classification_module/0816_loRA_fine_tuned_model_few"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float16,
    device_map="auto"  # GPU 자동 할당
)

# === 프롬프트 생성 ===
def make_messages(title_content: str, candidate_labels: list):
    return [
        {
            "role": "system",
            "content": """You are a helpful assistant that classifies documents into the most appropriate label based on their content.

- You must choose **only one** label from the given candidate list.
- Do not make up any new label that is not in the candidate list.
- Your answer must be in **Korean only**.
- Do not include explanations, reasoning, or phrases like "Answer:" — just output the label itself.

Here are some examples:

---

Document: 고기에서 대장균이 검출되었다는 보고가 있습니다.  
Candidate labels: 미생물일반, 대장균, 노로바이러스  
Output: 대장균

---

Document: 시료에서 살모넬라균이 검출되어 회수 조치가 진행 중입니다.  
Candidate labels: 대장균, 살모넬라, 곰팡이수  
Output: 살모넬라

---

Document: 해당 제품은 유통기한이 지난 후 진균수가 기준치를 초과했습니다.  
Candidate labels: 곰팡이수, 진균수, 황색포도상구균  
Output: 진균수

---

Now, classify the following document. Respond with **only one Korean label from the candidates**.
""",
        },
        {
            "role": "user",
            "content": f"""
Document: {title_content}

Candidate labels: {', '.join(candidate_labels)}

Output:""",
        },
    ]

# === 예측 함수 ===
def predict_label_with_pipe(title_content, candidate_labels):
    messages = make_messages(title_content, candidate_labels)
    outputs = pipe(
        messages,
        max_new_tokens=128,
    )
    # Llama 3는 출력 형식이 list[dict], dict["generated_text"] 안에 있음
    generated = outputs[0]["generated_text"]
    # 마지막 응답의 content 추출
    response = generated[-1]["content"] if isinstance(generated, list) else generated

    del messages, outputs
    torch.cuda.empty_cache()
    gc.collect()
    return response.strip()

# === 시드 고정 ===
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# === 결과 저장 경로 ===
save_path = "/home/yena/Classification_module/trained_llama3-3b_few_valdata.csv"

# === 중복 처리 방지 ===
if os.path.exists(save_path):
    processed = pd.read_csv(save_path, encoding="utf-8-sig")
    already_done = set(processed["제목+내용"].tolist())
else:
    pd.DataFrame(columns=["제목+내용", "대분류", "중분류", "예측 소분류"]).to_csv(save_path, index=False, encoding="utf-8-sig")
    already_done = set()

# === 메인 루프 ===
for _, row in tqdm(total_data.iterrows(), total=total_data.shape[0]):
    if row["제목+내용"] in already_done:
        continue
    category = row["대분류"]
    subcategory = row["중분류"]
    question_text = row["제목+내용"]

    filtered_df = classification_data[
        (classification_data["대분류"] == category) &
        (classification_data["중분류"] == subcategory)
    ]
    subcategory_list = filtered_df["소분류"].dropna().tolist()
    predicted_label = predict_label_with_pipe(question_text, subcategory_list)

    result_row = {
        "제목+내용": question_text,
        "대분류": category,
        "중분류": subcategory,
        "예측 소분류": predicted_label,
    }
    pd.DataFrame([result_row]).to_csv(save_path, mode='a', index=False, header=False, encoding="utf-8-sig")

    torch.cuda.empty_cache()
    gc.collect()
