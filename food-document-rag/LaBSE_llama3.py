from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import faiss
import random
import accelerate
import pickle
import json
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 3"

# 엑셀 읽기
file_path = "/home/yena/Food_RAG/식품안전정보DB-url 추가(2014~2023).xls"
df = pd.read_excel(file_path, sheet_name="2023", usecols=["제목", "내용"])
df["제목_내용"] = df["제목"] + " " + df["내용"]
data = df["제목_내용"].to_list()

# 문장 임베딩용 LaBSE
labse_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
labse_model = AutoModel.from_pretrained("sentence-transformers/LaBSE")

def mean_pooling(model_output, attention_mask):
    """Mean Pooling 적용 (패딩 토큰 제외)"""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())

    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)  # 패딩 부분 제외하고 평균 계산
    return sum_embeddings / sum_mask

def embed_texts(text_list):
    """LaBSE 모델을 사용하여 Mean Pooling 적용한 문장 임베딩을 생성하는 함수"""
    embeddings_list = []
    
    for text in tqdm(text_list):
        inputs = labse_tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        
        with torch.no_grad():
            outputs = labse_model(**inputs)
        
        sentence_embedding = mean_pooling(outputs, inputs['attention_mask'])
        embeddings_list.append(sentence_embedding)
    
    return torch.stack(embeddings_list).squeeze(1)

def embed_texts_CLS(text_list, model, tokenizer):
    """CLS 토큰을 사용한 문장 임베딩 생성 (진행률 표시)"""
    embeddings_list = []
    
    for text in text_list:
        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS 토큰의 임베딩
        embeddings_list.append(cls_embedding)
    
    return torch.stack(embeddings_list).squeeze(1)

def prompting_answer(question, data):
    messages = [
        {'role': 'system', 'content': '당신은 주어진 정보를 바탕으로 질문에 대해 구체적이고 상세한 답변을 제공하는 한국어 전문가입니다. 반드시 한국어로 답변하세요.'},
        {'role': 'user', 'content': '당신의 역할은 주어진 정보를 바탕으로 질문에 대해 구체적이고 자세한 답변을 제공하는 것입니다. 정보는 리스트 형식으로 주어집니다. 정확하고 상세한 답변이 요구됩니다.'},
        {"role": "assistant", "content": "물론입니다! 질문이 무엇인지 알려주시면, 최선을 다해 답변드리겠습니다."},
        {"role": "user", "content": f'''
            정보 : {data}
            질문 : {question}
            (비고, 다음의 조건들을 충족하도록 답변하세요.
            - 반드시 한국어로 작성
            - 구체적이고 상세한 답변 제공
            - 정답이 여러 개인 경우, 한 개만 선택하고 선택 이유를 간략히 포함)
            '''},
        {"role": "assistant", "content": '정답:'}
    ]
    return messages

# embeddings = embed_texts(data)
# with open("/home/yena/Food_RAG/labse_embeddings.pkl", "wb") as f:
#     pickle.dump(embeddings, f)

file_path = "/home/yena/Food_RAG/labse_embeddings.pkl"
with open(file_path, 'rb') as file:  # 'rb'는 바이너리 읽기 모드
    embeddings = pickle.load(file)
embeddings

# 1. 질의 리스트
query = [
    "Waitrose 8 Red Onion Bhajis with a Date and Tamarind Dip 제품이 왜 회수되었나?",
    "Hassui Kamaboko Co.,Ltd.에서 회수한 제품의 회수 사유는?",
    "프랑스 Ducourau사의 굴이 회수된 이유는?",
    "말레이시아 트렝가누주에서 발생한 식중독 사망 원인은?",
    "Arauco 올리브유가 판매금지된 이유는?",
    "벨기에에서 Isla Délice 식육가공품이 회수된 이유는?",
    "영국 Country Kitchen에서 회수한 머핀의 제품명은?",
    "Le Duo des Gors 치즈가 회수된 이유는?",
    "뉴질랜드에서 Value brand 탄산음료가 회수된 이유는?",
    "미국에서 TGD Cuts, LLC가 회수한 과일의 오염 가능성이 있는 병원균은?",
    "FishMeatz LLP가 벌금형을 받은 이유는 무엇인가?",
    "Rude Health Organic Coconut Drink가 회수된 이유는 무엇인가?",
    "New Roots Herbal의 아슈와간다 제품이 회수된 이유는 무엇인가?",
    "벨기에에서 BERGERONNETTE Pérail du Fédou 치즈가 회수된 이유는 무엇인가?",
    "루샤오자 식품 유한공사의 오향 닭날개가 부적합 판정을 받은 이유는 무엇인가?",
    "GDE Grocery Delivery E-Services Canada Inc.가 회수한 닭고기 제품의 유통지역은 어디인가?",
    "이탈리아에서 수출한 신선 여름 송로버섯의 카드뮴 함량은 얼마인가?",
    "Mrs Kirkham 치즈가 회수된 이유는 무엇인가?",
    "코스타리카 내 리스테리아증 감염 사례는 주로 어떤 식품과 관련이 있는가?",
    "쓰촨 촨라오라오 식품 과학기술유한공사의 식용 식물 혼합유에서 검출된 부적합 물질은 무엇인가?",
    "베트남 하노이 시장관리국은 어떤 불법 행위를 적발했는가?",
    "일본 농림수산성이 칠레산 가금육 등의 수입중지 조치를 해제한 이유는 무엇인가?",
    "최근 세계동물보건기구(WOAH)가 보고한 고병원성 조류인플루엔자(H5N1) 발생 현황은 어떠한가?",
    "칠레에서 보고된 조류인플루엔자 A(H5) 인체 감염 사례의 주요 내용은 무엇인가?",
    "스위스 연방평의회는 화학물질 및 폐기물 협약 강화와 관련하여 어떤 조치를 취하고 있는가?",
    "벨기에 연방보건부는 아스파탐의 1일허용섭취량(ADI)을 변경하지 않은 이유는 무엇인가?",
    "미국 연구진의 연구에 따르면, 카드뮴 식이 노출이 가장 높은 연령대와 주요 노출 식품은 무엇인가?",
    "일본 유한회사 밸런스가 마들렌 제품을 회수한 이유와 해당 제품의 판매 정보는 무엇인가?",
    "중국 해관총서와 농업농촌부는 터키에서 발생한 가성우역의 유입을 방지하기 위해 어떤 조치를 시행했는가?",
    "대만 신베이시에서 적발된 가짜 양고기 판매 사건의 주요 내용은 무엇인가?",
    "Springbank Cheese Co.와 Le Grand Fromage에서 회수된 치즈 제품은 무엇이며, 회수 사유는 무엇인가요?",
    "스페인 식품안전영양청(AESAN)은 아일랜드산 자숙 냉동게에 대한 경고를 왜 철회했나요?",
    "영국 환경식품농촌부(Defra)가 안내한 국경 목표운영모델(Border Target Operating Model)은 무엇인가?",
    "나이지리아산 히비스커스 꽃에서 검출된 미승인 물질은 무엇인가?",
    "일본에서 회수된 '팔도 꼬꼬면'의 회수 사유는 무엇인가?",
    "대만 식약서가 일본산 수입식품의 방사능 검사를 중단한 품목은 무엇인가?",
    "네팔에서 보고된 H5N2 고병원성 조류 인플루엔자의 발생 규모는 어떻게 되는가?",
    "미국 식품의약품청이 'PrimeZen Black 6000' 제품에 대해 경고한 이유는 무엇인가?",
    "독일 CVUA가 서양 송로버섯을 포함한 제품에서 집중적으로 모니터링하는 이유는 무엇인가?",
    "미국 식품안전검사국이 공중보건경보를 발령한 냉동 닭고기 제품의 제조사는 어디인가?"
]

print("질문 개수:", len(query))  # 확인용 출력

# 2. 임베딩 불러오기
embeddings = torch.load("labse_embeddings.pt")  # torch tensor로 저장된 파일
embeddings = np.array(embeddings, dtype=np.float32)

# 3. FAISS 인덱스 생성 및 임베딩 추가
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
faiss.normalize_L2(embeddings)
index.add(embeddings)

def prompting_answer(question, data):
    messages = [
        {'role': 'system', 'content': '당신은 주어진 정보를 바탕으로 질문에 대해 구체적이고 상세한 답변을 제공하는 한국어 전문가입니다. 반드시 한국어로 답변하세요.'},
        {'role': 'user', 'content': '당신의 역할은 주어진 정보를 바탕으로 질문에 대해 구체적이고 자세한 답변을 제공하는 것입니다. 정보는 리스트 형식으로 주어집니다. 정확하고 상세한 답변이 요구됩니다.'},
        {"role": "assistant", "content": "물론입니다! 질문이 무엇인지 알려주시면, 최선을 다해 답변드리겠습니다."},
        {"role": "user", "content": f'''
            정보 : {data}
            질문 : {question}
            (비고, 다음의 조건들을 충족하도록 답변하세요.
            - 반드시 한국어로 작성
            - 구체적이고 상세한 답변 제공
            - 정답이 여러 개인 경우, 한 개만 선택하고 선택 이유를 간략히 포함)
            '''},
        {"role": "assistant", "content": '정답:'}
    ]
    return messages

# GPU 디바이스 설정 (0번 GPU로 설정)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 모델 경로
model_path = '/SSL_NAS/concrete/models/models--meta-llama--Meta-Llama-3-8B-Instruct/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e5e23bbe8e749ef0efcf16cad411a7d23bd23298'

# 모델 로드 (low_cpu_mem_usage 제거)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,   # GPU 메모리 효율화
    device_map="auto"            # 자동으로 최적의 장치에 로드
)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# 5. 시드 고정 함수
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

# 7. 각 쿼리마다 Top-K 검색 및 답변 생성
k = 10  # top-k 유사 질문 개수
results = []
for q in tqdm(query):
    # 1. 질의 임베딩
    query_embedding = embed_texts([q])
    query_embedding = query_embedding.cpu().numpy().astype(np.float32)  
    faiss.normalize_L2(query_embedding)

    # 2. Top-K 유사 질문 검색
    distances, indices = index.search(query_embedding, k)
    top_10_embeddings = embeddings[indices[0]] 
    top_k_context = [data[i] for i in indices[0]]

    # 3. 프롬프트 구성 및 모델 입력
    messages = prompting_answer(q, top_k_context)
    templated_inputs = tokenizer.apply_chat_template(messages, tokenize=False)
    model_inputs = tokenizer(templated_inputs, padding=True, truncation=True, max_length=3500, return_tensors='pt').to(device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=300)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    answer = {q: output.split('Answer:')[-1].split('\n\n')[-1]}

    # 5. 결과 저장
    result = {
        "query": q,
        "top_k": top_k_context,
        "answer": answer
    }
    results.append(result)

    # 6. 콘솔 출력
    print(f"\n🟡 질문: {q}")
    print("🔹 Top-K 유사 질문:")
    for i, sim_q in enumerate(top_k_context, 1):
        print(f"   {i}. {sim_q}")
    print(f"✅ 답변: {answer}")

# 7. JSON 파일로 저장
with open("rag_query_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n✅ 모든 질문에 대한 답변이 저장되었습니다: rag_query_results.json")

# 8. GPU 자원 해제
del model
del tokenizer
del embeddings
del model_inputs
torch.cuda.empty_cache()  # GPU 메모리 초기화
gc.collect()              # 가비지 컬렉션