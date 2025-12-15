import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

# 1. 환경 설정
load_dotenv()
st.set_page_config(page_title="만능 AI 튜터", page_icon="🎓")
st.title("🎓 무엇이든 가르쳐 드려요!")

# 2. PDF 처리 및 벡터 DB 생성
@st.cache_resource
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(), docs

# 3. 요약 및 퀴즈 생성 (범용 버전)
def generate_summary_and_quiz(docs, topic, level):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    
    max_pages = 3
    context_text = "\n\n".join([doc.page_content for doc in docs[:max_pages]])

    # 프롬프트에 '주제(topic)'와 '대상(level)'을 동적으로 넣습니다.
    prompt = f"""
    너는 지금부터 유능한 '{topic}' 선생님이야.
    내 학생의 수준은 '{level}'이야. 이 수준에 딱 맞춰서 설명해야 해.
    
    아래 [교재 내용]을 보고 수업을 준비해줘.
    
    [교재 내용]:
    {context_text}
    
    [요청 사항]:
    1. **오늘의 핵심 요약**: 이 교재의 핵심 내용을 3가지로 요약해줘.
    2. **맞춤형 퀴즈**: '{level}' 수준에 맞는 3지 선다형 퀴즈를 1개 만들어줘.
    
    출력 형식:
    ---
    ### 📝 {topic} 핵심 요약 ({level}용)
    (요약 내용)
    
    ### 🧩 팝 퀴즈!
    (문제)
    1. (보기)
    2. (보기)
    3. (보기)
    ---
    """
    response = llm.invoke(prompt)
    return response.content

# --- UI 구성 ---
with st.sidebar:
    st.header("학습 설정 ⚙️")
    
    # [New] 과목명과 난이도를 사용자가 직접 고르게 합니다.
    topic = st.text_input("공부할 주제를 입력하세요", value="일반 상식")
    level = st.selectbox("학습 난이도(대상)", ["초등학생", "중고등학생", "대학생/전문가", "일반인"])
    
    st.divider()
    st.header("교재 업로드 📤")
    uploaded_file = st.file_uploader("PDF 교재를 올려주세요", type=["pdf"])

if uploaded_file is not None:
    temp_pdf_path = "temp_lesson.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        # 파일이 바뀌거나 설정이 바뀌면 리셋하기 위해 session_state 체크 로직을 단순화했습니다.
        if "retriever" not in st.session_state or st.sidebar.button("설정 적용 및 다시 학습"):
            with st.spinner(f"AI가 '{topic}' 과목을 '{level}' 수준으로 공부하는 중... 📚"):
                retriever, docs = process_pdf(temp_pdf_path)
                st.session_state["retriever"] = retriever
                
                # 요약본 생성 시 설정값 전달
                summary = generate_summary_and_quiz(docs, topic, level)
                
                st.session_state["messages"] = [
                    AIMessage(content=f"안녕하세요! 저는 오늘 여러분의 **{topic}** 선생님입니다.\n**{level}** 눈높이에 맞춰 수업할게요! 😎\n\n{summary}")
                ]
        st.success("준비 완료!")
        
    except Exception as e:
        st.error(f"오류 발생: {e}")
        st.stop()
else:
    # 파일 없으면 초기화
    for key in ["retriever", "messages"]:
        if key in st.session_state:
            del st.session_state[key]
    st.info("👈 왼쪽에서 주제를 적고 PDF를 업로드해주세요.")
    st.stop()

# 채팅 기록 표시
for msg in st.session_state["messages"]:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# 사용자 입력 처리
if user_input := st.chat_input("궁금한 점을 물어보세요!"):
    st.chat_message("user").write(user_input)
    st.session_state["messages"].append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("교재 내용 찾아보는 중..."):
            retriever = st.session_state["retriever"]
            retrieved_docs = retriever.invoke(user_input)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # 대화 기록 (기억력)
        chat_history = []
        for msg in st.session_state["messages"][-3:]: 
            role = "AI 선생님" if isinstance(msg, AIMessage) else "학생"
            chat_history.append(f"{role}: {msg.content}")
        history_text = "\n".join(chat_history)

        # [New] 프롬프트에도 topic과 level을 주입하여 페르소나 유지
        prompt_template = ChatPromptTemplate.from_template(f"""
        너는 유능하고 친절한 '{topic}' 선생님이야.
        학습자는 '{level}' 수준이야. 어려운 말 쓰지 말고 눈높이에 맞춰서 설명해.
        
        [지시 사항]:
        1. [대화 기록]을 참고해서 문맥을 이어가.
        2. 반드시 [참고 자료]에 기반해서 대답해. 모르면 모른다고 해.
        3. 설명은 명확하고 친절하게.

        [대화 기록]:
        {{history}}

        [참고 자료]:
        {{context}}
        
        질문: {{input}}
        """)
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
        chain = prompt_template | llm
        
        chunks = chain.stream({
            "history": history_text, 
            "context": context, 
            "input": user_input
        })
        
        for chunk in chunks:
            if chunk.content:
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.03)
        
        message_placeholder.markdown(full_response)
    
    st.session_state["messages"].append(AIMessage(content=full_response))