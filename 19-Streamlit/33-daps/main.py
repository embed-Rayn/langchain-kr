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


def add_history(role, content, idx=-1):
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
# try:
#     with open("examples.json", "r", encoding="utf8") as f: # 추후 DB로 대체
#         examples = json.load(f)
#     prompt = FewShotPromptTemplate(
#             examples=examples,
#             example_prompt=example_prompt,
#             suffix=(
#                 "Question:\n{question}\n"
#                 "Provide the answer and a sentiment score (1-10) in JSON format."
#             ),
#             input_variables=["question"],
#         )
# except FileNotFoundError:
#     pass
st.session_state["chain"] = create_chain(prompt, "gpt-4.1-2025-04-14")  # 체인(LLM+프롬프트+파서) 생성 및 세션에 저장

print_history()  # 이전 대화 기록 출력

if user_input := st.chat_input():  # 사용자 입력이 있으면
    add_history("user", user_input)  # 사용자 메시지 히스토리에 추가
    st.chat_message("user").write(user_input)  # 사용자 메시지 화면에 출력
    with st.chat_message("assistant"):
        chat_container = st.empty()  # AI 답변 실시간 표시용 컨테이너
        stream_response = st.session_state["chain"].stream({"question": user_input})  # LLM 체인에 질문 전달(스트리밍)
        ai_result_text = ""
        for chunk in stream_response:
            try:
                ai_answer = chunk['answer']  # 스트림에서 answer 추출
                chat_container.markdown(ai_answer)  # 실시간 답변 표시
            except KeyError:
                pass
            sentiment = 0  # 기본 감정점수
        sentiment = chunk["sentiment"]  # 마지막 chunk에서 sentiment 추출
        print(ai_answer)
        print(sentiment)
        add_history("ai", ai_answer)  # AI 답변 히스토리에 추가
        
        # 생성된 답변 표시와 함께 수정 버튼 생성 (열 분할 사용)
        st.markdown("**생성된 답변:**")
        st.write(ai_answer)
        col1, col2 = st.columns([3,1])  # 3:1 비율로 열 분할
        with col1:
            st.empty()  # 왼쪽 열은 빈 공간
        with col2:
            if st.button("답변 수정", key="edit_answer"):
                st.session_state["edit_answer"] = ai_answer  # 답변 수정 버튼 클릭 시, 세션에 답변 저장

        # "답변 수정" 버튼이 눌렸으면 text_area를 생성하여 수정 가능하도록 함.
        if "edit_answer" in st.session_state:
            edited_answer = st.text_area("답변 수정 하기", value=st.session_state["edit_answer"], key="edited_answer")
            if st.button("수정 적용", key="apply_edit"):
                ai_answer = edited_answer  # 수정된 답변으로 교체
                st.session_state["edit_answer"] = ai_answer
                st.success("수정된 답변이 적용되었습니다.")
                add_history("ai", ai_answer)  # 수정된 답변 히스토리에 추가

    # 예시: chain 실행 후 받는 결과를 활용하여 새 항목 생성
    new_entry = {
        "question": user_input,
        "answer": ai_answer,
        "sentiment": sentiment
    }

    # 기존의 examples.json 파일 읽고 새 항목 추가 후 저장
    try:
        with open("examples.json", "r", encoding="utf8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []  # 파일이 없다면 빈 리스트 생성

    data.append(new_entry)  # 새 항목 추가

    with open("examples.json", "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)  # 파일에 저장
