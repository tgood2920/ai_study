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
st.set_page_config(page_title="RFP 입찰 분석기 (Pro)", page_icon="📑", layout="wide")

# [추가할 코드] ★★★ 여기가 중요합니다! ★★★
# 대화 기록 사물함이 없으면 미리 빈 통을 만들어둡니다.
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    
st.title("📑 제안요청서(RFP) 핵심 분석기")
st.markdown("복잡한 공고문, **30초 만에 핵심만 파악**하고 **독소 조항**을 찾아냅니다.")

# 2. PDF 처리 (기존과 동일)
@st.cache_resource
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(), docs

# 3. [핵심] RFP 분석 전문 프롬프트
def analyze_rfp(docs, project_name):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3) # 분석은 창의성보다 정확성이 중요하므로 temperature를 낮춤
    
    # 앞부분(공고 개요)과 뒷부분(평가 기준)을 골고루 보기 위해 앞 5페이지 + 뒤 3페이지 정도를 조합하면 좋지만, 
    # 일단 심플하게 앞부분 5페이지만 읽어서 개요를 파악하게 합니다. (전체 분석은 RAG로 질문)
    max_pages = 5
    context_text = "\n\n".join([doc.page_content for doc in docs[:max_pages]])

    prompt = f"""
    너는 공공사업 입찰 및 제안서 작성 전문가(Senior PM)야.
    내가 업로드한 [제안요청서(RFP)]의 앞부분 내용을 바탕으로 아래 항목들을 아주 명확하게 정리해줘.
    
    [분석 대상 사업명]: {project_name}
    
    [RFP 내용 일부]:
    {context_text}
    
    [요청 사항 - 반드시 아래 포맷으로 출력]:
    
    ## 1. 🎯 사업 개요 요약
    * **사업 목적**: (한 줄 요약)
    * **사업 예산**: (금액이 보이면 적고, 안 보이면 '문서 내 검색 필요'라고 적음)
    * **사업 기간**: (기간 명시)
    * **주요 과업**: (핵심 요구사항 3~5가지 불렛 포인트)

    ## 2. ⚠️ 리스크 및 제약사항 (독소 조항 체크)
    * **입찰 자격**: (특정 라이선스나 실적 요구가 있는지)
    * **인력 요건**: (PM등급, 상주 여부 등 특이사항)
    * **패널티/제약**: (지체상금률, 기술료 등 위험 요소가 보이면 기술)

    ## 3. 📝 제안서 목차 추천 (초안)
    (이 RFP에 맞춰서 우리가 작성해야 할 제안서의 목차(Index)를 1, 2, 3단계로 구성해줘)
    
    ---
    **💡 Tip:** 더 자세한 기술 요구사항이나 평가 항목은 채팅창에 물어보시면 찾아드릴게요!
    """
    response = llm.invoke(prompt)
    return response.content

# --- UI 구성 ---
with st.sidebar:
    st.header("📂 분석 파일 업로드")
    
    project_name = st.text_input("사업명 (프로젝트 이름)", value="차세대 정보시스템 구축 사업")
    
    st.info("💡 HWP 파일은 PDF로 변환해서 올려주세요.")
    uploaded_file = st.file_uploader("RFP(PDF) 파일을 올려주세요", type=["pdf"])
    
    st.divider()
    st.markdown("### 🤖 사용 팁")
    st.markdown("""
    - **파일 업로드** 시 자동 분석이 시작됩니다.
    - 분석 후 **채팅창**에 이렇게 물어보세요.
        - "평가 기준표 보여줘"
        - "서버 구축 요구사항이 뭐야?"
        - "제출 서류 목록 정리해줘"
    """)

if uploaded_file is not None:
    # 1. 파일 이름에 원래 이름을 붙여서 고유하게 만듭니다.
    temp_pdf_path = f"temp_rfp_{uploaded_file.name}"
    
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 2. 파일이 바뀌었는지 체크 (없으면 생성, 다르면 초기화)
    if "last_uploaded_file" not in st.session_state or st.session_state["last_uploaded_file"] != uploaded_file.name:
        st.session_state["last_uploaded_file"] = uploaded_file.name
        st.session_state["messages"] = []      # 대화 기록 초기화
        if "retriever" in st.session_state:
            del st.session_state["retriever"]  # 기존 학습 내용 삭제

    try:
        # 3. 분석 시작 (retriever가 없을 때만 실행)
        if "retriever" not in st.session_state:
            with st.spinner(f"🔍 '{project_name}' 제안요청서를 꼼꼼히 분석 중입니다..."):
                
                retriever, docs = process_pdf(temp_pdf_path)
                st.session_state["retriever"] = retriever
                
                # 분석 결과 생성
                analysis_result = analyze_rfp(docs, project_name)
                
                # [안전장치] 혹시 모르니 여기서도 append 하기 전에 리스트 확인
                if "messages" not in st.session_state:
                    st.session_state["messages"] = []

                st.session_state["messages"].append(
                    AIMessage(content=f"**[{project_name}]** 분석이 완료되었습니다. 핵심 내용은 아래와 같습니다. 👇\n\n{analysis_result}")
                )
                
        st.success("분석 완료! 채팅으로 상세 내용을 물어보세요.")
        
    except Exception as e:
        st.error(f"오류 발생: {e}")
        st.stop()

# 채팅 인터페이스
for msg in st.session_state["messages"]:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant", avatar="👨‍💼").write(msg.content) # 아바타를 직장인으로 변경

# 사용자 질문 처리
if user_input := st.chat_input("예: 기술 평가 항목이 뭐야? / 투입 인력 조건이 있어?"):
    st.chat_message("user").write(user_input)
    st.session_state["messages"].append(HumanMessage(content=user_input))

    with st.chat_message("assistant", avatar="👨‍💼"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("제안요청서에서 관련 조항 찾는 중... 📑"):
            retriever = st.session_state["retriever"]
            retrieved_docs = retriever.invoke(user_input)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # 채팅용 프롬프트 (전문가 페르소나)
        prompt_template = ChatPromptTemplate.from_template(f"""
        너는 제안서 작성 전문가(PM)야. 
        사용자는 이 사업을 수주하고 싶어 하는 제안 담당자야.
        
        [지시 사항]:
        1. [참고 자료]인 제안요청서 내용에 근거해서 팩트 위주로 답변해.
        2. 제안서 작성에 도움이 되는 팁(전략)을 한 줄씩 덧붙여주면 더 좋아.
        3. 문서에 없는 내용은 "RFP에 명시되지 않았습니다"라고 솔직하게 말해.

        [참고 자료]:
        {{context}}
        
        담당자 질문: {{input}}
        """)
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        chain = prompt_template | llm
        
        chunks = chain.stream({"context": context, "input": user_input})
        
        for chunk in chunks:
            if chunk.content:
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.03)
        
        message_placeholder.markdown(full_response)
    
    st.session_state["messages"].append(AIMessage(content=full_response))