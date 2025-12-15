import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 환경변수 로드
load_dotenv()

# 1. 모델 준비
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.7
)

# 2. 대화 기록(Memory) 저장소 초기화
# [Web Dev 개념] 마치 DB에서 초기 채팅 로그를 불러온 것과 같습니다.
# SystemMessage: 개발자가 설정한 '페르소나' (사용자는 못 봄)
messages = [
    SystemMessage(content="너는 초등학생들을 가르치는 친절한 로봇 선생님 '코디'야. 🤖 반말은 쓰지 말고 해요체로 다정하게 말해줘.")
]

print("🤖 코디 선생님이 깨어났어요! (종료하려면 '그만' 이라고 입력하세요)")
print("-" * 30)

# 3. 무한 루프 (Game Loop / Server Loop)
while True:
    # 사용자 입력 받기
    user_input = input("나: ")
    
    # 종료 조건
    if user_input == "그만":
        print("코디: 그럼 안녕~ 다음에 또 만나! 👋")
        break

    # 4. 사용자 메시지를 기록에 추가 (Push)
    # [Web Dev 개념] 프론트엔드에서 보낸 메시지를 대화 리스트에 push() 하는 것과 동일
    messages.append(HumanMessage(content=user_input))

    try:
        # 5. 지금까지의 모든 대화 기록(messages)을 통째로 AI에게 전달
        # (AI는 이 리스트를 읽고 문맥을 파악한 뒤 다음 말을 생성함)
        response = llm.invoke(messages)
        
        # 6. AI의 답변 출력
        print(f"코디: {response.content}")
        
        # 7. AI 메시지도 기록에 추가 (Push)
        # 다음 턴에서 AI가 자기가 했던 말을 기억하게 하기 위함
        messages.append(AIMessage(content=response.content))
        
    except Exception as e:
        print(f"에러 발생: {e}")