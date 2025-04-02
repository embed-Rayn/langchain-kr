"""
LangChain을 활용한 상품 리뷰 응답 생성 시스템

이 코드는 다음 기능을 제공합니다:
1. 상품 리뷰의 감정 분석 (긍정/부정/중립)
2. CS 상황 분류 (환불/배송/교환/상품 품질/일반 피드백)
3. 상황별 맞춤형 응답 생성

필요한 패키지:
- langchain
- langchain-openai
- python-dotenv
"""

import os
from typing import Dict, List, Optional, Tuple

# LangChain 관련 임포트
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from langchain.memory import ConversationBufferMemory
from langchain.chains import SequentialChain

# 환경 설정
def setup_environment():
    """
    환경 변수 설정 및 필요한 디렉토리 생성
    """
    # OpenAI API 키 설정 (실제 사용 시 .env 파일이나 환경 변수에서 로드하는 것이 안전합니다)
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    print("환경 설정이 완료되었습니다.")

# 감정 분석을 위한 출력 파서 정의
class SentimentAnalysisOutput(BaseModel):
    """리뷰 감정 분석 결과"""
    sentiment: str = Field(description="리뷰의 감정 (positive, negative, neutral 중 하나)")
    confidence: float = Field(description="감정 분석의 확신도 (0.0 ~ 1.0)")
    key_points: List[str] = Field(description="리뷰에서 언급된 주요 포인트 목록")

# CS 상황 분류를 위한 출력 파서 정의
class CSSituationOutput(BaseModel):
    """CS 상황 분류 결과"""
    situation_type: str = Field(description="CS 상황 유형 (refund, shipping, exchange, product_quality, general_feedback 중 하나)")
    urgency_level: str = Field(description="긴급도 (high, medium, low 중 하나)")
    specific_issues: List[str] = Field(description="구체적인 문제점 목록")

# 감정 분석기 생성
def create_sentiment_analyzer(llm):
    """
    리뷰의 감정을 분석하는 체인 생성
    """
    parser = PydanticOutputParser(pydantic_object=SentimentAnalysisOutput)
    
    sentiment_template = """
    당신은 상품 리뷰의 감정을 분석하는 전문가입니다.
    아래 리뷰를 분석하여 감정(긍정, 부정, 중립)을 판단하고, 주요 포인트를 추출해주세요.
    
    리뷰: {review}
    
    {format_instructions}
    """
    
    sentiment_prompt = PromptTemplate(
        template=sentiment_template,
        input_variables=["review"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    sentiment_chain = LLMChain(
        llm=llm,
        prompt=sentiment_prompt,
        output_key="sentiment_analysis",
        verbose=True
    )
    
    return sentiment_chain, parser

# CS 상황 분류기 생성
def create_cs_situation_classifier(llm):
    """
    리뷰에서 CS 상황을 분류하는 체인 생성
    """
    parser = PydanticOutputParser(pydantic_object=CSSituationOutput)
    
    situation_template = """
    당신은 고객 리뷰에서 CS 상황을 분류하는 전문가입니다.
    아래 리뷰를 분석하여 상황 유형(환불, 배송, 교환, 상품 품질, 일반 피드백)을 판단하고, 
    긴급도와 구체적인 문제점을 추출해주세요.
    
    리뷰: {review}
    감정 분석 결과: {sentiment_analysis}
    
    {format_instructions}
    """
    
    situation_prompt = PromptTemplate(
        template=situation_template,
        input_variables=["review", "sentiment_analysis"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    situation_chain = LLMChain(
        llm=llm,
        prompt=situation_prompt,
        output_key="situation_classification",
        verbose=True
    )
    
    return situation_chain, parser

# 응답 생성기 생성
def create_response_generator(llm):
    """
    상황에 맞는 응답을 생성하는 체인 생성
    """
    # 상황별 프롬프트 템플릿 정의
    response_templates = {
        "refund": """
        당신은 친절하고 전문적인 고객 서비스 담당자입니다.
        아래 환불 관련 리뷰에 대해 공감하고 해결책을 제시하는 응답을 작성해주세요.
        
        리뷰: {review}
        감정 분석: {sentiment_analysis}
        상황 분류: {situation_classification}
        
        다음 사항을 포함해주세요:
        1. 고객의 불편에 대한 진심 어린 사과
        2. 환불 정책 안내
        3. 환불 절차 설명
        4. 추가 질문이 있을 경우 연락 방법
        
        응답:
        """,
        
        "shipping": """
        당신은 친절하고 전문적인 고객 서비스 담당자입니다.
        아래 배송 관련 리뷰에 대해 공감하고 해결책을 제시하는 응답을 작성해주세요.
        
        리뷰: {review}
        감정 분석: {sentiment_analysis}
        상황 분류: {situation_classification}
        
        다음 사항을 포함해주세요:
        1. 배송 지연이나 문제에 대한 이해와 사과
        2. 현재 배송 상태 확인 방법
        3. 문제 해결을 위한 구체적인 조치
        4. 향후 개선 약속
        
        응답:
        """,
        
        "exchange": """
        당신은 친절하고 전문적인 고객 서비스 담당자입니다.
        아래 교환 관련 리뷰에 대해 공감하고 해결책을 제시하는 응답을 작성해주세요.
        
        리뷰: {review}
        감정 분석: {sentiment_analysis}
        상황 분류: {situation_classification}
        
        다음 사항을 포함해주세요:
        1. 교환이 필요한 상황에 대한 이해와 사과
        2. 교환 정책 안내
        3. 교환 절차 설명
        4. 교환 과정에서 발생할 수 있는 비용이나 시간
        
        응답:
        """,
        
        "product_quality": """
        당신은 친절하고 전문적인 고객 서비스 담당자입니다.
        아래 상품 품질 관련 리뷰에 대해 공감하고 해결책을 제시하는 응답을 작성해주세요.
        
        리뷰: {review}
        감정 분석: {sentiment_analysis}
        상황 분류: {situation_classification}
        
        다음 사항을 포함해주세요:
        1. 품질 문제에 대한 진심 어린 사과
        2. 품질 관리 프로세스 간략 설명
        3. 문제 해결을 위한 구체적인 조치
        4. 품질 개선을 위한 피드백 감사 표현
        
        응답:
        """,
        
        "general_feedback": """
        당신은 친절하고 전문적인 고객 서비스 담당자입니다.
        아래 일반 피드백 리뷰에 대해 감사하고 적절한 응답을 작성해주세요.
        
        리뷰: {review}
        감정 분석: {sentiment_analysis}
        상황 분류: {situation_classification}
        
        다음 사항을 포함해주세요:
        1. 피드백에 대한 감사 표현
        2. 긍정적인 피드백이라면 기쁨 표현, 부정적이라면 개선 약속
        3. 고객의 의견이 중요하다는 메시지
        4. 향후 서비스/제품 개선에 반영하겠다는 약속
        
        응답:
        """
    }
    
    # 응답 생성 체인 생성
    response_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            template="{template}",
            input_variables=["template"]
        ),
        output_key="response",
        verbose=True
    )
    
    return response_chain, response_templates

# LangChain 파이프라인 구현
def create_review_response_pipeline():
    """
    전체 리뷰 응답 파이프라인 생성
    """
    # LLM 모델 초기화
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
    
    # 각 컴포넌트 생성
    sentiment_chain, sentiment_parser = create_sentiment_analyzer(llm)
    situation_chain, situation_parser = create_cs_situation_classifier(llm)
    response_chain, response_templates = create_response_generator(llm)
    
    # 전체 파이프라인 구성
    def process_review(review: str) -> Dict:
        """
        리뷰를 처리하고 적절한 응답 생성
        """
        # 감정 분석
        sentiment_result = sentiment_chain.run(review=review)
        try:
            parsed_sentiment = sentiment_parser.parse(sentiment_result)
            sentiment_dict = parsed_sentiment.dict()
        except Exception as e:
            print(f"감정 분석 결과 파싱 오류: {e}")
            sentiment_dict = {"sentiment": "neutral", "confidence": 0.5, "key_points": ["파싱 오류"]}
        
        # 상황 분류
        situation_result = situation_chain.run(review=review, sentiment_analysis=sentiment_result)
        try:
            parsed_situation = situation_parser.parse(situation_result)
            situation_dict = parsed_situation.dict()
        except Exception as e:
            print(f"상황 분류 결과 파싱 오류: {e}")
            situation_dict = {"situation_type": "general_feedback", "urgency_level": "low", "specific_issues": ["파싱 오류"]}
        
        # 상황에 맞는 템플릿 선택
        situation_type = situation_dict.get("situation_type", "general_feedback")
        template = response_templates.get(situation_type, response_templates["general_feedback"])
        
        # 템플릿에 변수 채우기
        filled_template = template.format(
            review=review,
            sentiment_analysis=sentiment_result,
            situation_classification=situation_result
        )
        
        # 응답 생성
        response = response_chain.run(template=filled_template)
        
        # 결과 반환
        return {
            "review": review,
            "sentiment_analysis": sentiment_dict,
            "situation_classification": situation_dict,
            "response": response
        }
    
    return process_review

# 샘플 리뷰로 테스트
def test_with_sample_reviews(pipeline):
    """
    샘플 리뷰로 파이프라인 테스트
    """
    sample_reviews = [
        # 환불 관련 부정적 리뷰
        "이 제품 정말 실망스럽네요. 광고와 전혀 다르고 품질도 형편없어요. 당장 환불해주세요. 돈 아까워요.",
        
        # 배송 관련 부정적 리뷰
        "주문한 지 2주가 지났는데 아직도 배송이 안 왔어요. 고객센터에 전화해도 연결이 안 되고, 정말 화가 납니다.",
        
        # 교환 관련 중립적 리뷰
        "제품은 괜찮은데 사이즈가 생각보다 작네요. 한 사이즈 큰 걸로 교환하고 싶은데 절차가 어떻게 되나요?",
        
        # 상품 품질 관련 부정적 리뷰
        "사용한 지 일주일 만에 고장 났어요. 이런 저품질 제품을 이 가격에 파는 건 사기입니다. 품질 관리 좀 제대로 하세요.",
        
        # 일반 피드백 긍정적 리뷰
        "정말 만족스러운 구매였어요! 배송도 빠르고 제품 품질도 좋네요. 다음에도 이용할게요. 감사합니다!"
    ]
    
    results = []
    for review in sample_reviews:
        print(f"\n===== 리뷰 처리 시작 =====\n{review}\n")
        result = pipeline(review)
        results.append(result)
        print(f"\n----- 응답 -----\n{result['response']}\n")
    
    return results

# 메인 함수
def main():
    """
    메인 실행 함수
    """
    # 환경 설정
    setup_environment()
    
    # 파이프라인 생성
    pipeline = create_review_response_pipeline()
    
    # 샘플 리뷰로 테스트
    test_results = test_with_sample_reviews(pipeline)
    
    print("\n===== 모든 테스트 완료 =====")
    
    # 실제 사용 예시
    print("\n===== 실제 사용 예시 =====")
    print("아래 코드를 사용하여 실제 리뷰에 응답할 수 있습니다:")
    print("""
    # 파이프라인 생성
    pipeline = create_review_response_pipeline()
    
    # 리뷰 처리
    review = "고객의 실제 리뷰 내용"
    result = pipeline(review)
    
    # 응답 출력
    print(result['response'])
    """)

if __name__ == "__main__":
    main()