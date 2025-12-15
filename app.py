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
st.set_page_config(page_title="코디 선생님 (Live)", page_icon="👨‍🏫")
st.title("👨‍🏫 실시간으로 가르쳐주는 '코디'")

# 2. PDF 처리 및 벡터 DB 생성
@st.cache_resource
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    # 퀴즈 생성을 위해 검색기(retriever)와 원본(docs) 둘 다 반환
    return vectorstore.as_retriever(), docs

# 3. 요약 및 퀴즈 생성
def generate_summary_and_quiz(docs):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    
    # 앞부분 3페이지만 읽기
    max_pages = 3
    context_text = "\n\n".join([doc.page_content for doc in docs[:max_pages]])

    prompt = f"""
    너는 초등학생들의 눈높이에 맞춘 친절한 선생님 '코디'야. 
    아래 [교재 내용]을 보고 수업을 준비해줘.
    
    [교재 내용]:
    {context_text}
    
    [요청 사항]:
    1. **오늘의 학습 목표**: 핵심 주제 3가지를 뽑아서 요약해줘.
    2. **재미있는 퀴즈**: 아이들이 흥미를 가질만한 내용으로 3지 선다형 퀴즈를 1개만 만들어줘.
    
    출력 형식:
    ---
    ### 📝 오늘의 학습 목표
    (요약)
    
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
    st.header("교재 업로드 📤")
    uploaded_file = st.file_uploader("PDF 파일을 올려주세요", type=["pdf"])

if uploaded_file is not None:
    temp_pdf_path = "temp_lesson.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        if "retriever" not in st.session_state:
            with st.spinner("코디가 교재를 읽고 수업 준비 중... 📚"):
                retriever, docs = process_pdf(temp_pdf_path)
                st.session_state["retriever"] = retriever
                
                summary = generate_summary_and_quiz(docs)
                
                st.session_state["messages"] = [
                    AIMessage(content=f"자, 수업 시작해볼까요? 😎\n\n{summary}")
                ]
        st.success("수업 준비 완료!")
        
    except Exception as e:
        st.error(f"오류 발생: {e}")
        st.stop()
else:
    if "retriever" in st.session_state:
        del st.session_state["retriever"]
        st.session_state["messages"] = []
    st.info("👈 왼쪽에서 PDF 교재를 먼저 업로드해주세요.")
    st.stop()

# 채팅 기록 표시
for msg in st.session_state["messages"]:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# ★ 여기가 핵심: 사용자 입력 처리 및 스트리밍
if user_input := st.chat_input("질문해 보세요!"):
    # 1. 사용자 메시지 화면 표시
    st.chat_message("user").write(user_input)
    st.session_state["messages"].append(HumanMessage(content=user_input))

    # 2. AI 답변 생성
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # (1) 검색 단계
        with st.spinner("교과서 뒤적이는 중... 📖"):
            retriever = st.session_state["retriever"]
            retrieved_docs = retriever.invoke(user_input)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # (2) [New] 대화 기록 가져오기 (최근 3개만 가져와서 기억력 주입)
        # 너무 많이 가져오면 토큰 비용이 드니, 최근 대화(퀴즈 낸 거)만 가져옵니다.
        chat_history = []
        for msg in st.session_state["messages"][-3:]: 
            role = "AI 선생님" if isinstance(msg, AIMessage) else "학생"
            chat_history.append(f"{role}: {msg.content}")
        history_text = "\n".join(chat_history)

        # (3) 프롬프트 구성 (대화 기록 포함)
        prompt_template = ChatPromptTemplate.from_template(f"""
        너는 친절한 선생님 '코디'야. 
        
        [지시 사항]:
        1. 아래 [대화 기록]을 보고 흐름을 파악해. (네가 방금 퀴즈를 냈다면 정답을 확인해줘!)
        2. [참고 자료]에 없는 내용은 솔직히 모른다고 해.
        3. 학생의 질문이나 대답에 친절하게 반응해줘.

        [대화 기록]:
        {{history}}

        [참고 자료]:
        {{context}}
        
        학생 질문: {{input}}
        """)
        
        # (4) LLM 호출
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
        chain = prompt_template | llm
        
        # history와 context, input을 모두 넣어줍니다.
        chunks = chain.stream({
            "history": history_text, 
            "context": context, 
            "input": user_input
        })
        
        # (5) 스트리밍 출력
        for chunk in chunks:
            if chunk.content:
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.03) # 타자 속도
        
        message_placeholder.markdown(full_response)
    
    # 3. 세션에 저장
    st.session_state["messages"].append(AIMessage(content=full_response))