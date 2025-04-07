#기본 Fewshot

from langchain_openai import ChatOpenAI
from langchain_teddynote.messages import stream_response

# 객체 생성
llm = ChatOpenAI(
    temperature=0.1,  # 창의성
    model_name="gpt-4o-mini",  # 모델명 "gpt-4o-mini" vs "o3-mini"
)



from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


examples = [
	{
		"question": """다이어트중인 울애들 간식찾아보다가 선택한
페페로니 시리즈세트!
6종세트가 도착했는데 색감이 너무 이뻐서
울애기들 앉혀놓고 한컷찍엇어요 ㅋㅋㅋ
추가 샘플간식도 보내주셔서 감사해용
오자마자 샘플 뜯어줫는데 다먹은봉지까지 들고
냄새맡고 난리난리!!ㅋㅋㅋ
요즘 열심히 다이어트중인 우리아이들
낮에는 너무 더워 산책을못해서 집에 무기력하게
있는 아이들에게 노즈워크해주려고 다이어트라인
미역, 고구마부터 소주종이컵에넣어서 던져줫더니
아주 우리아이들 환장을해요!!! ㅋㅋㅋ
11살10살인 노령견인 우리아이들 치아에도
부담없을정도로 정말 말캉말캉하더라구요~
간식라인마다 오메가3 .프로바이틱스도 같이있어서
진짜 건강영양간식이네요 !!!
너무잘먹는 간식 앞으로 정착합니다!!!
""",
	    "answer": "소중한 후기 감사합니다! 앞으로도 건강하고 맛있는 간식으로 보답드릴게요 :)",
	    "sentiment": 10
	},
	{
		"question": """냥이 선물로 구매해봤습니당
구성이 좋아서 주문했는데 역시나 완전 만족스러워요 ㅋㅋ 포장도 깔끔하고 6종으로 다양하게 구성되어있어 좋습니다
냥이가 슬금슬금와서 애교를 부려대니 너무 자주줘서 문제긴해요..ㅋㅋㅋ
저희 냥이는 특히 치즈고구마를 제일 좋아하는듯합니다!
""",
	    "answer": "냥이도 만족했다니 저희도 기쁘네요! 치즈고구마 인기 많아요 :)",
	    "sentiment": 9
	},
	{
		"question": """아 너무나도 패키지가 귀엽고 예쁘고,
냥이 간식인데..색감도 예쁘고 맛나보여서 
제가 다 맛보고싶네요ㅎ 선물용으로 최고이고~
동생이 키워서 선물해줬는데..
사료속에 섞어줬더니 간식만 쏙쏙 잘빼먹는다네요!

서비스 간식 많이 챙겨주셔서 감사드리고..
사람도 그렇듯..영양소 골고루 챙겨먹어야는데,
반려묘에게 필요한 영양가있는 간식 골라서 챙겨줄수있어 좋은거같네요^^""",
    "answer": "정성 가득한 후기 감사드려요! 선물용으로도 만족하셨다니 뿌듯합니다 :)",
    "sentiment": 9
	},
	{
		"question": """소고기 연어 닭은 종종 먹였는데 미역이랑 치즈는
처음 구입했어요~ 안먹을까봐 걱정했는데 잘 먹네요^^ 유통기한은 내년 2월까지 몇개랑 7월까지랑 넉넉해요. 치즈 간식이 세모모양이라 너무 귀엽네요 😆""",
    "answer": "다양한 맛을 즐기고 있다니 다행이에요! 귀여운 모양도 인기 포인트랍니다 :)",
    "sentiment": 8
	},
	{
		"question": """약먹일때나 노즈워크 간식으로 딱이에요""",
	    "answer": "활용도 높게 잘 써주셔서 감사합니다! 앞으로도 만족 드릴게요 :)",
    	"sentiment": 8
	},
	{
		"question": """두세트삿는데 입맛까다로운데 잘먹네요^^""",
	    "answer": "입맛 까다로운 아이도 잘 먹는다니 정말 기뻐요! 감사합니다 :)",
    	"sentiment": 8
	},
	{
		"question": """10년차 말티즈 두마리 견주로서 여러 간식들을 먹여 봤는데요! 이 간식은 가히 최고라하지 않을 수 없습니다. 장점 나열해 보겠습니다.

1.기호성은 동영상으로 첨부했는데 아주아주 좋아하네요... 맛도 가리질 않아요.
2.가성비 좋은 가격. 저는 강세일 때 사긴 했으나 세일 안 한 가격도 적절하다 생각해요.
3.포장 예쁨. 솔직히 덜렁 간식 오는 것보다 종이박스에 6종 담겨오니 견주인 저도 기분 좋네요.
4.적당히 말랑말랑함.노견들도 아주 잘 먹을 수 있는 경도예요. 솔직히 개껌같은 건 이제 제가 무서워서 못 먹이는데 이건 저희 할머니들에게 아주 적합한 정도네요.
5.다양한 맛 구성. 간식 여러 군데서 사기 귀찮은데 참 소비자들의 수요를 잘 아시는 듯 합니다.

이거 다 먹이면 또 살거예요~강세일로 좋은 브랜드 알게돼서 너무 좋습니다! 중간에 먹이다가 사진 다시 놓고 찍어서 용량 적어보이는 통이 있는데 원래는 꽉 차 있었어요~ 용량 가지고도 장난 안 치셔서 좋습니다~""",
	    "answer": "견주님의 세심한 후기에 감사드립니다! 아이들에게도 만족스러웠다니 더없이 기쁩니다. 다음에도 좋은 간식으로 찾아뵐게요!",
    	"sentiment": 10
	},
	{
		"question": """수분감이 많은 트윗이라 맨손으로 쪼개 주기도 너무 좋아요. 통 하나 하나가 작아서 양이 안 많아 보였는데, 먹다 보니 은근 오래 먹이네요 빈 공간 없이 꽉 차 있어서 그런가 봐요""",
	    "answer": "편하게 나눠주실 수 있고, 양도 충분했다니 정말 다행이에요! 감사합니다 :)",
    	"sentiment": 9
	},
	{
		"question": """다 좋은데 포장 상태가 영 별루입니다. 상자로 판매하시는 의미가 있는건지....  뽁뽁이라도 상자에 둘러서 보내주셨어야 겉 상자가 손상되지않았을것 같은데.. 여기저기 찌그러진걸 보니... 이걸 어찌 선물해야하나싶고. 그리고 리뷰쓰려고 보니 저는 35000원을 지불하고 샀는데 가격이 그새 만원이나 저렴하게 할인을하나요? 더 기분 나빠요 ...물건도 저상태. 가격도 이상태.""",
	    "answer": "불편을 드려 정말 죄송합니다. 포장 개선과 가격정책에 대해 내부적으로 꼭 공유하여 개선하겠습니다.",
    	"sentiment": 3
	},
	{
		"question": """선물용으로 재구매 했는데, 아무리 그래도 그렇지.. 케이스가 다 찌그러질 정도로 뽁뽁이 하나 없이 비닐 포장만 해서 보내는건 좀 아닌듯... 시간이 없어 교환은 안했는데 개선 좀 해주세요.""",
	    "answer": "포장 관련 불편을 드려 정말 죄송합니다. 앞으로는 선물용으로도 만족하실 수 있도록 더욱 신경 쓰겠습니다.",
    	"sentiment": 4
	}
]

example_prompt = PromptTemplate.from_template(
    "Question:\n{question}\nAnswer:\n{answer}"
)

# print(example_prompt.format(**examples[0]))

# prompt = FewShotPromptTemplate(
#     examples=examples,
#     example_prompt=example_prompt,
#     suffix="Question:\n{question}\nAnswer:",
#     input_variables=["question"],
# )

# question = "애들이 잘 먹긴하는데 빨간 눈물이 생겨서 급여를 못하고있어요ㅠㅠ"
# question = "매번 주문하는 간식인데 배송이 잘못되서 직접 찾을수가 받아왔습니다\n박스는 뜯어놔서 없고 내용물은 있어서 그나마 다행"
# question = "헤헹 너무 맛있게 잘 먹어요. 저도 하나 먹어봤는데 슴슴하니 맛있음"
# final_prompt = prompt.format(question=question)
# print(final_prompt)

# # 결과 출력
# answer = llm.stream(final_prompt)
# stream_response(answer)

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question:\n{question}\nAnswer:",
    input_variables=["question"],
)

# chain 생성
chain = prompt | llm | StrOutputParser()

# 결과 출력
# answer = chain.stream(
#     {"question": "애들이 잘 먹긴하는데 빨간 눈물이 생겨서 급여를 못하고있어요ㅠㅠ"}
# )
# stream_response(answer)


answer = chain.stream(
    {"question": "매번 주문하는 간식인데 배송이 잘못되서 직접 찾을수가 받아왔습니다\n박스는 뜯어놔서 없고 내용물은 있어서 그나마 다행"}
)
stream_response(answer)

# answer = chain.stream(
#     {"question": "헤헹 너무 맛있게 잘 먹어요. 저도 하나 먹어봤는데 슴슴하니 맛있음"}
# )
# stream_response(answer)
