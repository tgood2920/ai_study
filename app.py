import streamlit as st
import os
import time
import pandas as pd
import io
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

# 1. 초기 설정
load_dotenv()
st.set_page_config(page_title="RFP 분석 & 스토리보드", page_icon="📑", layout="wide")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

st.title("📑 RFP 분석 및 제안 스토리보드 생성기")

# 2. PDF 처리 함수
@st.cache_resource
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(), docs

# 3. 기본 분석 함수 (중소기업기술정보진흥원 명칭 준수)
def analyze_rfp(docs, project_name):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    context_text = "\n\n".join([doc.page_content for doc in docs[:10]])
    prompt = f"""너는 전문 PM이야. 아래 RFP를 분석해. 
    반드시 '중소기업기술정보진흥원' 명칭을 그대로 사용하고 줄이지 마.
    사업명: {project_name}\n내용: {context_text}
    양식: ## 1. 개요, 2. 리스크(독소조항), 3. 제안전략"""
    return llm.invoke(prompt).content

# 4. 스토리보드 데이터 생성 (사용자 엑셀 양식 반영)
def generate_storyboard_data(docs, project_name):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
    context_text = "\n\n".join([doc.page_content for doc in docs[:15]])
    
    prompt = f"""너는 제안 기획자야. RFP를 토대로 '제안목차' 시트 데이터를 만들어.
    사업명: {project_name}\n내용: {context_text}
    
    [출력 규칙]:
    1. 반드시 파이프(|)로 구분된 CSV 형식으로만 답해. 설명은 생략해.
    2. 컬럼명: 목차|작성자|요구사항ID|작성 지침(필수)|평가배점|평가기준
    3. 목차는 I.II.1.1.1 순서로 상세히 구성해.
    """
    return llm.invoke(prompt).content

# --- UI 영역 ---
with st.sidebar:
    st.header("⚙️ 설정")
    project_name = st.text_input("사업명", value="사업명을 입력하세요")
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
        if st.button("📊 엑셀 스토리보드 생성"):
            with st.spinner("엑셀 파일 구성 중..."):
                raw_data = generate_storyboard_data(st.session_state["docs"], project_name)
                try:
                    # 데이터 클리닝: 불필요한 설명 및 마크다운 제거
                    clean_data = re.sub(r'^[^{|]*\n', '', raw_data, flags=re.MULTILINE)
                    lines = [l.strip() for l in clean_data.split('\n') if '|' in l]
                    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|")
                    
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False, sheet_name='5. 제안목차_변경')
                        # 엑셀 서식 자동 지정
                        ws = writer.sheets['5. 제안목차_변경']
                        for i, col in enumerate(df.columns):
                            ws.set_column(i, i, 25)
                    
                    st.success("생성 완료!")
                    st.dataframe(df)
                    st.download_button("📥 엑셀 다운로드", output.getvalue(), f"{project_name}_스토리보드.xlsx")
                except:
                    st.error("데이터 파싱 실패. AI가 형식을 지키지 않았습니다. 다시 시도해주세요.")
                    st.text_area("AI 응답 원본", raw_data)

    if user_input := st.chat_input("질문하세요"):
        st.chat_message("user").write(user_input)
        st.session_state["messages"].append(HumanMessage(content=user_input))
        with st.chat_message("assistant"):
            ctx = "\n".join([d.page_content for d in st.session_state["retriever"].invoke(user_input)])
            ans = ChatGoogleGenerativeAI(model="gemini-2.0-flash").invoke(f"RFP내용: {ctx}\n질문: {user_input}").content
            st.write(ans)
            st.session_state["messages"].append(AIMessage(content=ans))
else:
    st.info("RFP PDF를 업로드해 주세요.")