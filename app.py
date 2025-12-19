import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage

# 외부 모듈 로드
from excel_gen import get_basic_info_json, create_basic_info_excel

load_dotenv()
st.set_page_config(page_title="RFP 분석 시스템", page_icon="📑", layout="wide")

# 세션 초기화
for key in ["messages", "retriever", "docs", "analysis_done"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "messages" else None if key != "analysis_done" else False

st.title("📑 RFP 분석 엔진")

@st.cache_resource
def load_and_analyze_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(), docs

# --- 사이드바 및 업로드 ---
with st.sidebar:
    st.header("📂 분석 도구")
    project_alias = st.text_input("프로젝트 명칭", "입찰_사업")
    uploaded_file = st.file_uploader("RFP PDF 업로드", type=["pdf"])
    if st.button("🗑️ 초기화"):
        st.session_state.clear()
        st.rerun()

if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())

    if st.session_state["retriever"] is None:
        with st.spinner("📄 PDF를 정밀 분석 중입니다..."):
            retriever, docs = load_and_analyze_pdf(temp_path)
            st.session_state.update({"retriever": retriever, "docs": docs, "analysis_done": True})
            st.session_state["messages"].append(AIMessage(content="PDF 분석이 완료되었습니다."))

    # 채팅 UI
    for msg in st.session_state["messages"]:
        st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

    # 엑셀 생성 로직 (분리된 모듈 호출)
    if st.session_state["analysis_done"]:
        st.divider()
        if st.button("📊 1. 사업기본정보 엑셀 생성"):
            with st.spinner("스토리보드 양식에 맞춰 구성 중..."):
                data_dict = get_basic_info_json(st.session_state["docs"])
                excel_file = create_basic_info_excel(data_dict, project_alias)
                
                if excel_file:
                    st.success("1번 시트 생성이 완료되었습니다.")
                    st.download_button("📥 엑셀 다운로드", excel_file, f"{project_alias}_사업기본정보.xlsx")
                else:
                    st.error("데이터 추출에 실패했습니다.")

    # 대화 처리
    if user_input := st.chat_input("RFP에 대해 질문하세요."):
        st.chat_message("user").write(user_input)
        st.session_state["messages"].append(HumanMessage(content=user_input))
        with st.chat_message("assistant"):
            search_res = st.session_state["retriever"].invoke(user_input)
            ctx = "\n".join([d.page_content for d in search_res])
            ans = ChatGoogleGenerativeAI(model="gemini-2.0-flash").invoke(
                f"중소기업기술정보진흥원 명칭을 준수하고 축약해서 답변하라.\n내용: {ctx}\n질문: {user_input}"
            ).content
            st.write(ans); st.session_state["messages"].append(AIMessage(content=ans))