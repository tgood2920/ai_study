import streamlit as st
import os
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

# 1. 환경 설정
load_dotenv()
st.set_page_config(page_title="RFP 입찰 분석기 Pro", page_icon="📑", layout="wide")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

st.title("📑 RFP 분석 및 엑셀 데이터 추출")

# 2. PDF 처리 로직
@st.cache_resource
def process_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        return vectorstore.as_retriever(), docs
    except Exception as e:
        st.error(f"PDF 로드 실패: {e}")
        return None, None

# 3. [핵심] 1. 사업기본정보 데이터 추출 함수
def extract_basic_info_secure(docs):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    # 사업 개요가 집중된 앞부분 15페이지 참조
    context = "\n\n".join([doc.page_content for doc in docs[:15]])
    
    prompt = f"""
    너는 공공기관 입찰 데이터 분석가야. 아래 [RFP 내용]에서 정보를 찾아 반드시 JSON 형식으로만 응답해.
    
    [준수 사항]:
    1. '중소기업기술정보진흥원'은 절대 줄이지 말고 전체 명칭을 사용해.
    2. 나머지 모든 정보는 최대한 간결하게 축약해서 작성해.
    3. JSON 데이터 외에 어떤 설명이나 텍스트도 포함하지 마.

    [JSON 키값]:
    - 공식사업명
    - 수요기관
    - 사업기간
    - 사업예산(VAT포함)
    - 공고일
    - 입찰방식

    [RFP 내용]:
    {context}
    """
    
    response = llm.invoke(prompt).content
    
    # 정규표현식으로 JSON 데이터만 추출 (강력한 필터링)
    try:
        # 중괄호 사이의 내용을 찾음
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str), response
        return None, response
    except Exception as e:
        return None, str(e)

# --- UI 영역 ---
with st.sidebar:
    st.header("⚙️ 프로젝트 설정")
    project_name = st.text_input("프로젝트명", value="입력해주세요")
    uploaded_file = st.file_uploader("RFP PDF 업로드", type=["pdf"])

if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 파일 변경 시 초기화
    if "last_file" not in st.session_state or st.session_state["last_file"] != uploaded_file.name:
        st.session_state.update({"last_file": uploaded_file.name, "messages": [], "analysis_done": False})
        if "retriever" in st.session_state: del st.session_state["retriever"]

    if "retriever" not in st.session_state:
        with st.spinner("RFP 데이터를 정밀 분석 중입니다..."):
            retriever, docs = process_pdf(temp_path)
            if retriever:
                st.session_state.update({"retriever": retriever, "docs": docs})
                st.session_state["analysis_done"] = True
                st.session_state["messages"].append(AIMessage(content="분석이 끝났습니다. 아래 버튼을 눌러 1번 시트를 생성하세요."))

    # 채팅창 표시
    for msg in st.session_state["messages"]:
        st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

    # 엑셀 생성 영역
    if st.session_state["analysis_done"]:
        st.divider()
        if st.button("📁 1. 사업기본정보 엑셀 생성"):
            with st.spinner("엑셀 데이터를 구성 중입니다..."):
                data_dict, raw_res = extract_basic_info_secure(st.session_state["docs"])
                
                if data_dict:
                    df = pd.DataFrame(list(data_dict.items()), columns=['항목', '내용'])
                    
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False, sheet_name='1. 사업기본정보')
                        ws = writer.sheets['1. 사업기본정보']
                        # 헤더 스타일링
                        fmt = writer.book.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
                        for col_num, value in enumerate(df.columns.values):
                            ws.write(0, col_num, value, fmt)
                        ws.set_column(0, 0, 20)
                        ws.set_column(1, 1, 60)
                    
                    st.success("데이터 추출 및 엑셀 구성 완료!")
                    st.table(df) # 미리보기
                    st.download_button("📥 엑셀 다운로드", output.getvalue(), f"{project_name}_기본정보.xlsx")
                else:
                    st.error("데이터 추출 실패. AI 응답을 확인하세요.")
                    with st.expander("디버깅 정보 (AI 응답 원본)"):
                        st.code(raw_res)

    # Q&A 처리
    if user_input := st.chat_input("RFP에 대해 질문하세요."):
        st.chat_message("user").write(user_input)
        st.session_state["messages"].append(HumanMessage(content=user_input))
        with st.chat_message("assistant"):
            ctx = "\n".join([d.page_content for d in st.session_state["retriever"].invoke(user_input)])
            ans = ChatGoogleGenerativeAI(model="gemini-2.0-flash").invoke(f"RFP내용: {ctx}\n질문: {user_input}").content
            st.write(ans)
            st.session_state["messages"].append(AIMessage(content=ans))
else:
    st.info("RFP PDF 파일을 업로드하면 분석을 시작합니다.")