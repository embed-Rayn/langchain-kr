from langchain_core.prompts import (
    get_template_variables,
    PromptTemplate,
    format_document,
)

# 1. 베이스 프롬프트 정의 (Jinja2 문법)
base_template = """
You are a customer support assistant for an e‑commerce store.

Product: {{ product_title }}

When a customer leaves a review, generate a helpful response.
"""

# 2. Few‑shot 예제 템플릿 정의
example_template = """
Q: {{ question }}
AI: {{ ai_answer }}
Human revised: {{ revised_answer }}
"""

# 3. 예제 템플릿에서 플레이스홀더(변수) 추출
vars_in_example = get_template_variables(example_template, "jinja2")
# vars_in_example == ["question", "ai_answer", "revised_answer"]

# 4. PromptTemplate 생성
prompt = PromptTemplate(
    template=base_template
             + "\n\nHere are some examples:\n{{ few_shot_examples }}\n\n"
             + "Now reply to this new review:\nReview: {{ review_text }}",
    input_variables=["product_title", "few_shot_examples", "review_text"],
)

# 5. 실제 실행 시, few‑shot 예제 데이터를 렌더링
few_shot_data = [
    {
        "question": "Battery life is too short.",
        "ai_answer": "We're sorry to hear that...",
        "revised_answer": "죄송합니다. 사용 환경에 따라 배터리 소모가 빠를 수 있습니다..."
    },
    {
        "question": "Love the design, but it's expensive.",
        "ai_answer": "Thank you for your feedback...",
        "revised_answer": "고객님, 디자인을 마음에 들어 해주셔서 감사합니다..."
    },
]

# Jinja2 포맷터로 각 예제를 문자열로 변환 후 합치기
rendered_examples = "\n".join(
    prompt.format(**example) for example in few_shot_data
)

# 6. 최종 프롬프트 문자열 생성
filled_prompt = prompt.format(
    product_title="UltraFast Charger",
    few_shot_examples=rendered_examples,
    review_text="장점은 빠른 충전, 단점은 발열이 심함."
)

print(filled_prompt)
print(type(filled_prompt))