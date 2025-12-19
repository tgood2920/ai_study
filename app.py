import streamlit as st
import pandas as pd
import io
import json
import re
from dotenv import load_dotenv

# 에러 방지를 위한 필수 임포트
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 생성한 모듈 임포트
from excel_gen1 import write_sheet1
from excel_gen2 import write_sheet2

load_dotenv()
st.set_page_config(page_title="RFP 분석기", layout="wide")

# 세션 상태 초기화
for key in ["messages", "retriever", "docs", "analysis_done"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "messages" else None if key != "analysis_done" else False

st.title("📑 RFP 통합 분석 시스템")

# PDF 처리 함수
@st.cache_resource
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
    vectorstore = FAISS.from_documents(splits, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    return vectorstore.as_retriever(), docs

# app.py 내 get_integrated_data 함수 수정
def get_integrated_data(docs):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    
    # 분석 범위: 앞부분(개요) 10페이지 + 뒷부분(제출안내) 20페이지 조합
    # 제출 서류는 보통 문서 끝부분에 나오므로 범위를 뒤쪽으로 넓혔습니다.
    context = "\n\n".join([doc.page_content for doc in docs[:10]]) + \
              "\n\n" + "\n\n".join([doc.page_content for doc in docs[-20:]])
    
    prompt = f"""
    RFP를 분석해서 입찰 및 제안서 제출 정보를 JSON으로 응답해. 
    내용은 핵심만 아주 짧게 축약해. 데이터가 없으면 빈 문자열("")을 넣어.

    구조: {{
        "basic_info": {{
            "basic": {{ "공식사업명":"", "공고번호":"", "수요기관":"", "사업예산":"", "사업기간":"", "입찰방식":"" }},
            "managers": [ {{ "소속":"", "성명":"", "연락처":"", "이메일":"" }} ],
            "issues": [ {{ "구분":"", "주요사항":"", "비고":"" }} ],
            "status": [ {{ "일자":"", "주요사항":"", "비고":"" }} ]
        }},
        "prep_docs": {{
            "project_name": "사업명",
            "sub_method": "제안서 제출 방식 (예: 온라인 제출)",
            "sub_copies": "제출 부수 (예: 정량1식, 정성1식)",
            "doc_list": [
                {{ "구분": "서류 분류(예: 입찰참가서류)", "제출서류": "서류 명칭", "확인사항": "비고/참고" }}
            ]
        }}
    }}
    내용: {context}
    """
    res = llm.invoke(prompt).content
    match = re.search(r'\{.*\}', res, re.DOTALL)
    return json.loads(match.group(0)) if match else None

# --- UI 섹션 ---
with st.sidebar:
    project_alias = st.text_input("프로젝트명", "사업명을 입력하세요.")
    uploaded_file = st.file_uploader("RFP PDF 업로드", type=["pdf"])

if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())

    if not st.session_state["analysis_done"]:
        with st.spinner("📄 PDF 분석 중..."):
            retriever, docs = process_pdf(temp_path)
            st.session_state.update({"retriever": retriever, "docs": docs, "analysis_done": True})

    if st.session_state["analysis_done"]:
        if st.button("📊 스토리보드 생성"):
            with st.spinner("데이터 추출 및 시트 구성 중..."):
                data = get_integrated_data(st.session_state["docs"])
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    wb = writer.book
                    t_fmt = wb.add_format({'bold': True, 'font_size': 12, 'bottom' : 2 })
                    h_fmt = wb.add_format({'bold': True, 'align': 'center', 'bg_color': '#F2F2F2', 'border': 1})
                    c_fmt = wb.add_format({'border': 1, 'text_wrap': True, 'valign': 'vcenter'})
                    
                    write_sheet1(wb, data, t_fmt, h_fmt, c_fmt)
                    write_sheet2(wb, data, t_fmt, h_fmt, c_fmt)
                
                st.download_button("📥 스토리보드 다운로드", output.getvalue(), f"{project_alias}_제안요약.xlsx")