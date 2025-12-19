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

# 1. 환경 설정 및 초기화
load_dotenv()
st.set_page_config(page_title="RFP 마스터 분석기", page_icon="📑", layout="wide")

for key in ["messages", "retriever", "docs", "analysis_done"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "messages" else None if key != "analysis_done" else False

st.title("📑 RFP 통합 분석 및 멀티 시트 엑셀 생성")

# 2. PDF 처리 함수
@st.cache_resource
def process_pdf_file(file_path):
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        return vectorstore.as_retriever(), docs
    except Exception as e:
        return None, str(e)

# 3. 데이터 추출 핵심 함수
def extract_data_for_sheets(docs):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    context = "\n\n".join([doc.page_content for doc in docs[:20]]) # 핵심 20페이지 참조
    
    # 공통 지침
    base_instruction = "중소기업기술정보진흥원 명칭은 절대 줄이지 마라. 모든 설명은 최대한 축약하라."

    # 시트 1: 사업기본정보
    p1 = f"{base_instruction}\nJSON으로 응답하라. 키: 공식사업명, 수요기관, 사업기간, 사업예산, 입찰방식\n내용: {context}"
    res1 = llm.invoke(p1).content
    
    # 시트 5: 제안목차
    p5 = f"{base_instruction}\nJSON 배열로 응답하라. 각 객체 키: 목차, 요구사항ID, 작성 지침\n내용: {context}"
    res5 = llm.invoke(p5).content
    
    return res1, res5

# --- UI 섹션 ---
with st.sidebar:
    st.header("⚙️ 분석 설정")
    project_name = st.text_input("프로젝트명", value="사업명을 입력하세요")
    uploaded_file = st.file_uploader("RFP PDF 업로드", type=["pdf"])

if uploaded_file:
    if "current_file" not in st.session_state or st.session_state["current_file"] != uploaded_file.name:
        st.session_state.update({"current_file": uploaded_file.name, "messages": [], "analysis_done": False, "retriever": None})
        
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())

    if st.session_state["retriever"] is None:
        with st.spinner("📄 RFP 분석 중..."):
            retriever, result_docs = process_pdf_file(temp_path)
            if retriever:
                st.session_state.update({"retriever": retriever, "docs": result_docs, "analysis_done": True})
                st.session_state["messages"].append(AIMessage(content="RFP 분석 완료. 이제 엑셀 생성이 가능합니다."))
            else: st.error(f"분석 실패: {result_docs}")

    for msg in st.session_state["messages"]:
        st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

    if st.session_state["analysis_done"]:
        st.divider()
        if st.button("🚀 통합 제안 엑셀 생성 (시트 1, 5)"):
            with st.spinner("AI가 멀티 시트 데이터를 구성 중입니다..."):
                r1, r5 = extract_data_for_sheets(st.session_state["docs"])
                try:
                    # JSON 파싱
                    data1 = json.loads(re.search(r'\{.*\}', r1, re.DOTALL).group(0))
                    data5 = json.loads(re.search(r'\[.*\]', r5, re.DOTALL).group(0))
                    
                    df1 = pd.DataFrame(list(data1.items()), columns=['항목', '내용'])
                    df5 = pd.DataFrame(data5)
                    
                    # 멀티 시트 엑셀 생성
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df1.to_excel(writer, index=False, sheet_name='1. 사업기본정보')
                        df5.to_excel(writer, index=False, sheet_name='5. 제안목차_변경')
                        
                        # 서식 설정 (1번 시트)
                        ws1 = writer.sheets['1. 사업기본정보']
                        ws1.set_column(0, 0, 25); ws1.set_column(1, 1, 65)
                        
                        # 서식 설정 (5번 시트)
                        ws5 = writer.sheets['5. 제안목차_변경']
                        ws5.set_column(0, 0, 30); ws5.set_column(1, 1, 20); ws5.set_column(2, 2, 70)

                    st.success("통합 엑셀 생성 성공!")
                    st.download_button("📥 통합 엑셀 다운로드", output.getvalue(), f"{project_name}_통합제안서.xlsx")
                except:
                    st.error("데이터 생성 중 오류가 발생했습니다. 다시 시도해 주세요.")

    if user_input := st.chat_input("질문하세요."):
        st.chat_message("user").write(user_input)
        st.session_state["messages"].append(HumanMessage(content=user_input))
        with st.chat_message("assistant"):
            search_res = st.session_state["retriever"].invoke(user_input)
            ctx = "\n".join([d.page_content for d in search_res])
            ans = ChatGoogleGenerativeAI(model="gemini-2.0-flash").invoke(f"중소기업기술정보진흥원 명칭을 준수하고 축약해서 답변하라.\n내용: {ctx}\n질문: {user_input}").content
            st.write(ans); st.session_state["messages"].append(AIMessage(content=ans))
else: st.info("PDF를 업로드해 주세요.")