import json
from langchain_core.prompts import load_prompt
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DAPS 💬", page_icon="💬", layout="wide")
st.title("DAPS 💬")

def create_chain(prompt, model):
    llm = ChatOpenAI(
        temperature=0.1,
        model_name="gpt-4o-mini",
    )
    chain = prompt | llm | JsonOutputParser()
    return chain

prompt = load_prompt("prompts/default.yaml", encoding="utf8")
chain = create_chain(prompt, "gpt-4.1-2025-04-14")

# 데이터프레임 불러오기
df = pd.read_json("data/input_example.json", encoding="utf8")

# AI 답변(초안)과 감정점수 1회만 생성
if "draft_answers" not in st.session_state or "sentiments" not in st.session_state:
    # LangChain batch 실행
    inputs = [{"question": q} for q in df["question"].tolist()]
    results = list(chain.batch(inputs))
    draft_answers = []
    sentiments = []
    for res in results:
        draft_answers.append(res.get("answer", ""))
        sentiments.append(res.get("sentiment", 0))
    st.session_state["draft_answers"] = draft_answers
    st.session_state["sentiments"] = sentiments
else:
    draft_answers = st.session_state["draft_answers"]
    sentiments = st.session_state["sentiments"]

df["answer"] = draft_answers
df["sentiment"] = sentiments

# product_name = df["product_name"].iloc[0] if not df.empty else "제품명 없음"
product_name = "페페로니 간식 세트"
st.markdown(f"### {product_name} 리뷰 결과")

# 수정 답변을 session_state에 저장 (최초 1회만)
if "modified_answers" not in st.session_state:
    st.session_state["modified_answers"] = list(df["answer"])

header_cols = st.columns([1, 2, 10, 1, 8])
header_names = ["ID", "날짜", "리뷰", "감정점수", "수정 답변"]
for col, name in zip(header_cols, header_names):
    col.markdown(f"**{name}**")

for idx, row in df.iterrows():
    cols = st.columns([1, 2, 10, 1, 8])
    with cols[0]:
        st.markdown(f"{row['review_id']}")
    with cols[1]:
        st.markdown(f"{row['review_date']}")
    with cols[2]:
        st.markdown(f"{row['question']}".replace("~", "\~"))
    with cols[3]:
        st.markdown(f"{row['sentiment']}")
    with cols[4]:
        modified = st.text_area(
            "수정 답변",
            value=st.session_state["modified_answers"][idx],
            key=f"modified_answer_{idx}",
            height=80
        )
        st.session_state["modified_answers"][idx] = modified  # session_state에 저장

# 버튼 클릭 시 session_state의 값을 df에 반영
cols = st.columns([4, 1])
with cols[1]:
    if st.button("답변 수정 및 반영", key="edit_answer", type="primary"):
        changed_list = []
        for idx, row in df.iterrows():
            if row["answer"] != st.session_state["modified_answers"][idx]:
                changed_list.append({
                    "review_id": row["review_id"],
                    "question": row["question"],
                    "original_answer": row["answer"],
                    "modified_answer": st.session_state["modified_answers"][idx]
                })
        if changed_list:
            with open("data/changed_answers.json", "w", encoding="utf8") as f:
                json.dump(changed_list, f, ensure_ascii=False, indent=2)
            st.success("수정된 답변이 changed_answers.json 파일로 저장되었습니다.")
        else:
            st.info("변경된 답변이 없습니다.")
