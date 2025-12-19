import streamlit as st
import os
import time
import pandas as pd
import io
import json
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage

# 1. 초기 설정
load_dotenv()
st.set_page_config(page_title="RFP 분석기 Pro", page_icon="📑", layout="wide")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

st.title("📑 RFP 입찰 분석기 (사업기본정보 특화)")

# 2. PDF 처리
@st.cache_resource
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(), docs

# 3. RFP 기본 요약 분석
def analyze_rfp(docs, project_name):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    context = "\n\n".join([doc.page_content for doc in docs[:10]])
    prompt = f"""너는 공공입찰 전문가야. RFP를 분석해 핵심만 축약해서 보고해.
    **중소기업기술정보진흥원** 명칭을 절대 줄이지 마.
    사업명: {project_name}\n내용: {context}
    결과 항목: ## 1. 사업개요, 2. 리스크, 3. 핵심 요구사항"""
    return llm.invoke(prompt).content

# 4. [특화] 1. 사업기본정보 데이터 추출 (JSON 방식)
def extract_basic_info(docs):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    # 사업 개요가 집중된 앞부분 10페이지 참조
    context = "\n\n".join([doc.page_content for doc in docs[:10]])
    
    prompt = f"""RFP에서 다음 정보를 찾아 JSON 형식으로만 출력해. 
    **중소기업기술정보진험원** 명칭을 절대 줄이지 마. 모든 설명은 최대한 축약해.
    
    필수 항목: 
    - 공식사업명
    - 수요기관
    - 사업기간
    - 사업비용 (부가세 포함 여부 명시)
    - 공고일
    - 입찰방식
    
    [RFP 내용]: {context}
    
    출력 형식: {{ "공식사업명": "...", "수요기관": "...", "사업기간": "...", "사업비용": "...", "공고일": "...", "입찰방식": "..." }}
    """
    res = llm.invoke(prompt).content
    # JSON 문자열만 추출
    match = re.search(r'\{.*\}', res, re.DOTALL)
    return match.group(0) if match else "{}"

# --- UI 영역 ---
with st.sidebar:
    st.header("⚙️ 설정")
    project_name = st.text_input("프로젝트명", value="사업명을 입력하세요")
    uploaded_file = st.file_uploader("RFP PDF 업로드", type=["pdf"])

if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if "last_file" not in st.session_state or st.session_state["last_file"] != uploaded_file.name:
        st.session_state.update({"last_file": uploaded_file.name, "messages": [], "analysis_done": False})
        if "retriever" in st.session_state: del st.session_state["retriever"]

    if "retriever" not in st.session_state:
        with st.spinner("RFP 분석 중..."):
            retriever, docs = process_pdf(temp_path)
            st.session_state.update({"retriever": retriever, "docs": docs})
            res = analyze_rfp(docs, project_name)
            st.session_state["messages"].append(AIMessage(content=res))
            st.session_state["analysis_done"] = True

    for msg in st.session_state["messages"]:
        st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

    if st.session_state["analysis_done"]:
        st.divider()
        if st.button("📁 1. 사업기본정보 엑셀 생성"):
            with st.spinner("데이터 추출 중..."):
                json_raw = extract_basic_info(st.session_state["docs"])
                try:
                    data_dict = json.loads(json_raw)
                    # 데이터프레임 변환 (행태: 항목 | 내용)
                    df = pd.DataFrame(list(data_dict.items()), columns=['항목', '내용'])
                    
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False, sheet_name='1. 사업기본정보')
                        ws = writer.sheets['1. 사업기본정보']
                        ws.set_column(0, 0, 20)
                        ws.set_column(1, 1, 60)
                    
                    st.success("1. 사업기본정보 시트 생성 완료!")
                    st.table(df) # 깔끔한 표로 표시
                    st.download_button("📥 엑셀 다운로드", output.getvalue(), f"{project_name}_사업기본정보.xlsx")
                except:
                    st.error("데이터 파싱 실패. 다시 시도해주세요.")
                    st.write(json_raw)

    if user_input := st.chat_input("RFP에 대해 더 궁금한 점은?"):
        st.chat_message("user").write(user_input)
        st.session_state["messages"].append(HumanMessage(content=user_input))
        with st.chat_message("assistant"):
            ctx = "\n".join([d.page_content for d in st.session_state["retriever"].invoke(user_input)])
            ans = ChatGoogleGenerativeAI(model="gemini-2.0-flash").invoke(f"RFP내용: {ctx}\n질문: {user_input}").content
            st.write(ans)
            st.session_state["messages"].append(AIMessage(content=ans))
else:
    st.info("RFP PDF를 업로드하면 분석을 시작합니다.")