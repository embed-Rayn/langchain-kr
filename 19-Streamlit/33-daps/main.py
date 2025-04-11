import streamlit as st
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import ChatMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import load_prompt
from langchain_core.prompts.few_shot import FewShotPromptTemplate
import json


st.set_page_config(page_title="DAPS 💬", page_icon="💬")
st.title("DAPS 💬")

if "messages" not in st.session_state:
    st.session_state["messages"] = []


def print_history():
    for msg in st.session_state["messages"]:
        st.chat_message(msg.role).write(msg.content)


def add_history(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))


# 체인을 생성합니다.
def create_chain(prompt, model):
    llm = ChatOpenAI(
        temperature=0.1,  # 창의성
        model_name="gpt-4o-mini",  # 모델명 "gpt-4o-mini" vs "o3-mini"
    )
    # output을 JSON 형식으로 파싱하도록 변경
    chain = prompt | llm | JsonOutputParser()
    return chain

# 예제 템플릿: LLM이 answer와 sentiment를 JSON으로 응답하도록 함
example_prompt = PromptTemplate.from_template(
    '{"answer": "{answer}", "sentiment": {sentiment}}'
)

prompt = load_prompt("prompts/default.yaml", encoding="utf8")
try:
    with open("examples.json", "r", encoding="utf8") as f: # 추후 DB로 대체
        examples = json.load(f)
    prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            suffix=(
                "Question:\n{question}\n"
                "Provide the answer and a sentiment score (1-10) in JSON format."
            ),
            input_variables=["question"],
        )
except FileNotFoundError:
    pass
st.session_state["chain"] = create_chain(prompt, "o3-mini")

print_history()


if user_input := st.chat_input():
    add_history("user", user_input)
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        chat_container = st.empty()

        stream_response = st.session_state["chain"].stream({"question": user_input})
        ai_result_text = ""
        for chunk in stream_response:
            ai_result_text += chunk
            # 실시간으로 응답 진행 상황 표시
            chat_container.markdown(ai_result_text)
        # 스트리밍이 끝난 후 최종 결과 출력 및 파싱 시도
        print(f"ai_result_text: {ai_result_text}")
        try:
            result = json.loads(ai_result_text)
        except json.JSONDecodeError as e:
            print("JSON 파싱 오류:", e)
            result = {}
        ai_answer = result.get("answer", "")
        sentiment = result.get("sentiment", 0)
        add_history("ai", ai_answer)

    # 예시: chain 실행 후 받는 결과를 활용하여 새 항목 생성
    new_entry = {
        "question": user_input,
        "answer": ai_answer,      
        "sentiment": sentiment
    }

    # 기존의 examples.json 파일을 읽고 새 항목을 추가한 후 다시 저장합니다.
    try:
        with open("examples.json", "r", encoding="utf8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []  # 파일이 없다면 빈 리스트 생성

    data.append(new_entry)

    with open("examples.json", "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
