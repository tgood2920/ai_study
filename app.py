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

# 1. 초기 설정 및 보안
load_dotenv()
st.set_page_config(page_title="RFP 분석기 Pro", page_icon="📑", layout="wide")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

# 중소기업기술정보진흥원 명칭 준수 안내
st.title("📑 RFP 입찰 분석 및 스토리보드 생성기")

# 2. PDF 처리 로직
@st.cache_resource
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(), docs

# 3. RFP 분석 함수
def analyze_rfp(docs, project_name):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    context = "\n\n".join([doc.page_content for doc in docs[:10]])
    prompt = f"""너는 공공입찰 PM이야. 아래 RFP를 분석해. 
    **중소기업기술정보진흥원** 명칭을 절대 줄여 쓰지 마.
    사업명: {project_name}\n내용: {context}
    결과 항목: ## 1. 사업개요, 2. 리스크/독소조항, 3. 수주 전략"""
    return llm.invoke(prompt).content

# 4. 스토리보드 데이터 생성 (사용자 엑셀 양식 동기화)
def generate_storyboard_data(docs, project_name):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
    context = "\n\n".join([doc.page_content for doc in docs[:20]])
    
    prompt = f"""너는 제안서 기획자야. RFP를 분석해 '제안목차' 데이터를 생성해.
    **중소기업기술정보진흥원** 명칭을 준수해.
    
    [출력 규칙]:
    1. 반드시 파이프(|) 구분자를 사용한 CSV 형식으로만 출력해.
    2. 마크다운 코드 블록(```)이나 설명은 절대 포함하지 마.
    3. 헤더명: 목차|작성자|요구사항ID|작성 지침(필수)|평가배점|평가기준
    
    사업명: {project_name}\n내용: {context}"""
    return llm.invoke(prompt).content

# --- UI 영역 ---
with st.sidebar:
    st.header("⚙️ 설정")
    project_name = st.text_input("사업명", value="입력해주세요")
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
            with st.spinner("엑셀 시트 구성 중..."):
                raw = generate_storyboard_data(st.session_state["docs"], project_name)
                try:
                    # 데이터 정제 로직 강화: 마크다운 및 서술문 제거
                    clean = re.sub(r'```(?:csv|text)?|```', '', raw).strip()
                    lines = [l for l in clean.split('\n') if '|' in l]
                    
                    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|")
                    
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False, sheet_name='5. 제안목차_변경')
                        ws = writer.sheets['5. 제안목차_변경']
                        for i, col in enumerate(df.columns):
                            ws.set_column(i, i, 25) # 열 너비 조정
                    
                    st.success("스토리보드 생성 완료!")
                    st.dataframe(df)
                    st.download_button("📥 엑셀 다운로드", output.getvalue(), f"{project_name}_스토리보드.xlsx")
                except Exception as e:
                    st.error("파싱 오류 발생. AI의 응답 형식이 바르지 않습니다.")
                    st.text_area("원본 응답(디버깅용)", raw)

    if user_input := st.chat_input("질문하세요"):
        st.chat_message("user").write(user_input)
        st.session_state["messages"].append(HumanMessage(content=user_input))
        with st.chat_message("assistant"):
            ctx = "\n".join([d.page_content for d in st.session_state["retriever"].invoke(user_input)])
            ans = ChatGoogleGenerativeAI(model="gemini-2.0-flash").invoke(f"RFP내용: {ctx}\n질문: {user_input}").content
            st.write(ans)
            st.session_state["messages"].append(AIMessage(content=ans))
else:
    st.info("왼쪽에서 RFP PDF를 업로드해 주세요.")