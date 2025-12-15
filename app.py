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
    # 1. 사용자 메시지 표시
    st.chat_message("user").write(user_input)
    st.session_state["messages"].append(HumanMessage(content=user_input))

    # 2. AI 답변 생성 (스트리밍)
    with st.chat_message("assistant"):
        # 빈 박스를 먼저 만들어서 여기다가 글자를 하나씩 채울 겁니다.
        message_placeholder = st.empty()
        full_response = ""
        
        # (1) 검색 단계: 검색하는 동안은 스피너를 보여줍니다.
        with st.spinner("교과서 뒤적이는 중... 📖"):
            retriever = st.session_state["retriever"]
            retrieved_docs = retriever.invoke(user_input)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # (2) 생성 단계: 검색이 끝나면 바로 스트리밍 시작
        prompt_template = ChatPromptTemplate.from_template("""
        너는 친절한 선생님 '코디'야. 
        아래 [참고 자료]를 바탕으로 학생의 질문에 답변해줘.
        
        [참고 자료]:
        {context}
        
        질문: {input}
        """)
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
        chain = prompt_template | llm
        
        # ★ 강제 스트리밍 루프
        # chain.stream()에서 조각(chunk)이 나올 때마다 화면을 갱신합니다.
        chunks = chain.stream({"context": context, "input": user_input})
        
        for chunk in chunks:
            # content가 있는 경우에만 처리
            if chunk.content:
                full_response += chunk.content
                # ▌ 문자를 뒤에 붙여서 커서가 깜빡이는 느낌을 줍니다.
                message_placeholder.markdown(full_response + "▌")
                # 너무 빠르면 눈에 안 보일 수 있으니 아주 찰나의 딜레이 (선택사항)
                time.sleep(0.05)
        
        # 다 끝나면 커서(▌)를 없애고 최종본 확정
        message_placeholder.markdown(full_response)
    
    # 3. 세션에 저장
    st.session_state["messages"].append(AIMessage(content=full_response))