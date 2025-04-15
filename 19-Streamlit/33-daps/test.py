import random
import pandas as pd
import streamlit as st

df = pd.DataFrame(
    {
        "name": ["Roadmap", "Extras", "Issues"],
        "url": ["https://roadmap.streamlit.app", "https://extras.streamlit.app", "https://issues.streamlit.app"],
        "stars": [random.randint(0, 1000) for _ in range(3)],
        "views_history": [[random.randint(0, 5000) for _ in range(30)] for _ in range(3)],
    }
)
st.dataframe(
    df,
    column_config={
        "name": "App name",
        "stars": st.column_config.NumberColumn(
            "Github Stars",
            help="Number of stars on GitHub",
            format="%d ⭐",
        ),
        "url": st.column_config.LinkColumn("App URL"),
        "views_history": st.column_config.LineChartColumn(
            "Views (past 30 days)", y_min=0, y_max=5000
        ),
    },
    hide_index=True,
)

col1, col2 = st.columns([3, 1])
with col1:
    st.empty()  # 왼쪽 열은 빈 공간
with col2:
    if st.button("답변 수정 및 반영", key="edit_answer", primary=True):


"""
temp_df = df.copy().loc[:, ["review_id", "review_date", "question", "sentiment", "answer"]]
st.markdown("### 리뷰 결과 (수정 답변은 직접 입력)")

st.dataframe(
    temp_df,
    column_config={
        "review_id": st.column_config.TextColumn(
            "리뷰 ID",
            help="리뷰 고유 번호",
            disabled=True
        ),
        "review_date": st.column_config.TextColumn(
            "리뷰 날짜",
            help="YYYY-MM-DD",
            disabled=True
        ),
        "question": st.column_config.TextColumn(
            "리뷰 내용",
            help="고객 리뷰 원문",
            disabled=True
        ),
        "sentiment": st.column_config.NumberColumn(
            "감정점수",
            help="1~10점",
            min_value=1,
            max_value=10,
            format="%d",
            disabled=True
        ),
        "answer": st.column_config.TextColumn(
            "AI 답변",
            help="AI가 생성한 답변 (수정은 아래에서)",
            disabled=False
        ),
    },
    hide_index=True,
    use_container_width=True
)   """