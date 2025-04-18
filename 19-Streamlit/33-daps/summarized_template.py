from langchain_core.prompts import PromptTemplate, get_template_variables
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from typing import List, Dict
import json

# 1) 예제 템플릿 (ID, 날짜 포함)
example_template = """
Product: {{ product_title }}
Review ID: {{ review_id }}
Q: {{ question }}
AI: {{ original_answer }}
Human revised: {{ revised_answer }}
"""

# 2) 플레이스홀더 변수 파악 및 PromptTemplate 생성
vars_in_example = get_template_variables(example_template, "jinja2")
example_prompt = PromptTemplate(
    template=example_template,
    input_variables=vars_in_example,
    template_format="jinja2",  # ★ 이 부분 추가
)

def load_and_render_examples(reviews: List[Dict]) -> str:
    """
    reviews: List of dicts with keys:
        review_id, review_date, is_replied, question,
        ai_answer, revised_answer
    Returns: 각 미응답 리뷰에 대해 example_prompt를 적용해 합친 문자열
    """
    rendered = []
    for r in reviews:
        rendered.append(example_prompt.format(
            review_id=r["review_id"],
            question=r["question"],
            original_answer=r["original_answer"],
            revised_answer=r["revised_answer"]
        ))
    return "\n".join(rendered)

# 3) 사용 예시
def load_reviews_from_json(file_path: str) -> List[Dict]:
    """
    Reads a JSON file containing a list of review objects and returns it.
    Assumes the JSON is an array of objects with keys like
    'review_id', 'review_date', 'is_replied', 'question', etc.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        reviews: List[Dict] = json.load(f)
    return reviews


reviews_data = load_reviews_from_json("data/revised_answers.json")

rendered_examples = load_and_render_examples(reviews_data)
print(rendered_examples)

# 1) 요약용 프롬프트 만들기
summarize_template = """
당신은 대형 언어 모델의 전문가 프롬프트 작성자입니다.
당신의 목표는 제품 리뷰 답변에 대한 아래의 프롬프트를 개선하는 것입니다:
--------------------

예제들: {{ examples }}

--------------------

다음은 훌륭한 프롬프트를 작성하는 몇 가지 팁입니다:
-------
해당 주제에 대한 전문가임을 밝히며 프롬프트를 시작합니다.
프롬프트의 시작 부분에 지침을 넣고 ###을 사용하거나 지침과 문맥을 구분합니다
제품의 특징, 원하는 맥락, 결과, 길이, 형식, 스타일 등에 대해 구체적이고 설명적이며 가능한 한 자세히 설명해야 합니다
---------
다음은 훌륭한 프롬프트의 예입니다:
제품 마케팅, 고객 서비스 담당자로써 쇼핑몰에 달린 답변에 대한 답변을 생성하세요.
답변에는 제품 이름(Product)과 질문 내역(Q), 1차 답변 내역(AI), 2차 수정 답변 내역(Human revised)을 바탕으로 작성하세요.
2차 수정 답변의 수정된 부분의 특징을 파악하여 제품의 특성과 결합하여 프롬프트를 작성해야합니다.

예:
"당신은 20년차 Customer Service AI 어시스턴트 입니다. 사용자의 리뷰에 따라 적절한 리뷰를 작성해 주세요.
리뷰의 감정을 파악해서 감정 점수에 따라 긍정적일 수록 짧게, 부정적일 수록 길게 40~80자로 친절하게 작성하세요.
## 제품 특징:
1. 세모난 모양의 디자인
2. 다양한 구성"
...

## 말투:
1. 친절하고 정중하게
2. 적절한 이모지 사용

## header/footer:
1. header: "안녕하세요. XX입니다."
-----

이제 프롬프트를 개선하세요.
한국어로 작성합니다. 답변은 세 개의 따옴표로 감싸야 합니다.

개선된 프롬프트:

"""

summarize_prompt = PromptTemplate(
    template=summarize_template,
    input_variables=["examples"],
    template_format="jinja2"
)

# 2) 요약을 수행할 Runnable (LLMChain 역할)
def summarize_examples(examples_text):
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt_text = summarize_prompt.format(examples=examples_text)
    # LLM의 .invoke 또는 .predict 사용 (generate는 보통 여러 개 생성할 때)
    return llm.invoke(prompt_text).content

# 3) 기존 rendered_examples(길게 늘어선 few-shot 예제들) 를 요약
summary = summarize_examples(rendered_examples)
print("=== 핵심 패턴 요약 ===")
print(summary)
