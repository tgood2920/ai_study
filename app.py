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
st.set_page_config(page_title="RFP 입찰 분석기", page_icon="📑", layout="wide")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

st.title("📑 RFP 분석 및 사업기본정보 추출")

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

# 
# 3. [특화] 1. 사업기본정보 데이터 추출 함수
def extract_basic_info_json(docs):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    # 사업 개요가 집중된 앞부분 참조
    context = "\n\n".join([doc.page_content for doc in docs[:10]])
    
    prompt = f"""
    너는 공공입찰 데이터 추출 전문가야. 아래 RFP 내용을 분석해서 반드시 JSON 형식으로만 응답해.
    
    [필수 준수 사항]:
    1. '중소기업기술정보진흥원' 명칭은 절대 줄이지 말고 전체 이름을 사용해.
    2. 나머지 모든 설명과 내용은 최대한 간결하게 축약해서 표현해.
    3. JSON 외의 다른 설명이나 텍스트는 절대 포함하지 마.

    [추출 항목]:
    - 공식사업명
    - 수요기관
    - 사업기간
    - 사업예산(부가세포함)
    - 공고일
    - 입찰방식(계약방법)

    [RFP 내용]:
    {context}
    """
    
    response = llm.invoke(prompt).content
    
    # JSON 파싱을 위한 정규표현식 (중괄호 사이의 내용만 추출)
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return None
    except:
        return None

# --- UI 영역 ---
with st.sidebar:
    st.header("⚙️ 프로젝트 설정")
    project_name = st.text_input("프로젝트명", value="사업명을 입력하세요")
    uploaded_file = st.file_uploader("RFP PDF 업로드", type=["pdf"])

if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 파일 교체 시 초기화
    if "last_file" not in st.session_state or st.session_state["last_file"] != uploaded_file.name:
        st.session_state.update({"last_file": uploaded_file.name, "messages": [], "analysis_done": False})
        if "retriever" in st.session_state: del st.session_state["retriever"]

    if "retriever" not in st.session_state:
        with st.spinner("RFP 데이터를 분석 중입니다..."):
            retriever, docs = process_pdf(temp_path)
            st.session_state.update({"retriever": retriever, "docs": docs})
            st.session_state["analysis_done"] = True
            st.session_state["messages"].append(AIMessage(content="RFP 분석이 완료되었습니다. 아래 버튼을 눌러 기본정보 엑셀을 생성하세요."))

    # 채팅창 표시
    for msg in st.session_state["messages"]:
        st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

    # 엑셀 생성 영역
    if st.session_state["analysis_done"]:
        st.divider()
        if st.button("📁 1. 사업기본정보 엑셀 생성"):
            with st.spinner("데이터를 엑셀 형식으로 변환 중..."):
                data_dict = extract_basic_info_json(st.session_state["docs"])
                
                if data_dict:
                    # 데이터프레임 구성 (항목 | 내용)
                    df = pd.DataFrame(list(data_dict.items()), columns=['항목', '내용'])
                    
                    # 엑셀 파일 생성
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False, sheet_name='1. 사업기본정보')
                        
                        # 서식 설정
                        workbook = writer.book
                        worksheet = writer.sheets['1. 사업기본정보']
                        header_format = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
                        
                        for col_num, value in enumerate(df.columns.values):
                            worksheet.write(0, col_num, value, header_format)
                        worksheet.set_column(0, 0, 20)
                        worksheet.set_column(1, 1, 60)
                    
                    st.success("사업기본정보 추출 성공!")
                    st.table(df) # 화면에 표로 즉시 확인
                    st.download_button(
                        label="📥 엑셀 다운로드",
                        data=output.getvalue(),
                        file_name=f"{project_name}_사업기본정보.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.error("데이터 추출에 실패했습니다. 다시 시도해 주세요.")

    # 추가 질의응답
    if user_input := st.chat_input("RFP에 대해 더 궁금한 점을 물어보세요."):
        st.chat_message("user").write(user_input)
        st.session_state["messages"].append(HumanMessage(content=user_input))
        with st.chat_message("assistant"):
            ctx = "\n".join([d.page_content for d in st.session_state["retriever"].invoke(user_input)])
            ans = ChatGoogleGenerativeAI(model="gemini-2.0-flash").invoke(f"RFP내용: {ctx}\n질문: {user_input}").content
            st.write(ans)
            st.session_state["messages"].append(AIMessage(content=ans))
else:
    st.info("왼쪽 사이드바에서 PDF를 업로드하면 분석을 시작합니다.")