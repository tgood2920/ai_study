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
st.set_page_config(page_title="RFP 사업정보 추출기", page_icon="📝", layout="wide")

# 세션 상태 초기화 (안전성 확보)
for key in ["messages", "retriever", "docs", "analysis_done"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "messages" else None if key != "analysis_done" else False

st.title("📝 제안서 1. 사업기본정보 생성기")
st.caption("PDF에서 정보를 추출하여 1번 시트 양식에 맞춰 엑셀을 생성합니다.")

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

# 3. 데이터 추출 프롬프트 (중소기업기술정보진흥원 명칭 보존 및 축약 지침)
def extract_basic_info(docs):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    context = "\n\n".join([doc.page_content for doc in docs[:15]])
    
    prompt = f"""
    너는 공공입찰 전문가야. [RFP 내용]에서 정보를 찾아 JSON으로만 응답해.
    
    [규칙]:
    1. '중소기업기술정보진흥원' 명칭은 절대 줄이지 마라.
    2. 모든 내용은 최대한 글자수를 줄여서 축약하라.
    3. 데이터가 없으면 빈 문자열("")로 채워라.
    
    [JSON 키]:
    공식사업명, 공고번호, 수요기관, 사업예산(VAT포함), 사업기간, 입찰방식, 낙찰자결정방법, 입찰참가자격, 담당자정보

    [RFP 내용]:
    {context}
    """
    
    try:
        response = llm.invoke(prompt).content
        clean_json = re.search(r'\{.*\}', response, re.DOTALL).group(0)
        return json.loads(clean_json)
    except:
        return None

# --- UI 섹션 ---
with st.sidebar:
    st.header("⚙️ 설정")
    project_name = st.text_input("프로젝트 별칭", value="RFP_분석")
    uploaded_file = st.file_uploader("RFP PDF 업로드", type=["pdf"])
    if st.button("🗑️ 초기화"):
        st.session_state.clear()
        st.rerun()

if uploaded_file:
    if "current_file" not in st.session_state or st.session_state["current_file"] != uploaded_file.name:
        st.session_state.update({"current_file": uploaded_file.name, "messages": [], "analysis_done": False, "retriever": None})
        
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())

    if st.session_state["retriever"] is None:
        with st.spinner("📄 PDF 분석 중..."):
            retriever, result_docs = process_pdf_file(temp_path)
            if retriever:
                st.session_state.update({"retriever": retriever, "docs": result_docs, "analysis_done": True})
                st.session_state["messages"].append(AIMessage(content="분석 완료! 1번 시트를 생성할 수 있습니다."))
            else: st.error(f"분석 실패: {result_docs}")

    for msg in st.session_state["messages"]:
        st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

    if st.session_state["analysis_done"]:
        st.divider()
        if st.button("📊 1. 사업기본정보 시트 생성"):
            with st.spinner("엑셀 양식 구성 중..."):
                data_dict = extract_basic_info(st.session_state["docs"])
                
                if data_dict:
                    # 데이터프레임 생성 및 'None' 처리 (중요!)
                    df = pd.DataFrame(list(data_dict.items()), columns=['구분', '내용'])
                    df = df.fillna("") # 모든 NaN/None을 빈 문자열로 치환
                    
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False, sheet_name='1. 사업기본정보')
                        
                        workbook = writer.book
                        worksheet = writer.sheets['1. 사업기본정보']
                        
                        # 스타일 설정
                        header_fmt = workbook.add_format({'bold': True, 'align': 'center', 'bg_color': '#D7E4BC', 'border': 1})
                        content_fmt = workbook.add_format({'valign': 'vcenter', 'border': 1, 'text_wrap': True})
                        
                        # 데이터 쓰기 (TypeError 방지를 위해 강제 형변환)
                        for col_num, value in enumerate(df.columns.values):
                            worksheet.write(0, col_num, value, header_fmt)
                        
                        for row_num, row_data in enumerate(df.values):
                            worksheet.write(row_num + 1, 0, str(row_data[0]), content_fmt)
                            worksheet.write(row_num + 1, 1, str(row_data[1]), content_fmt)
                        
                        worksheet.set_column(0, 0, 25)
                        worksheet.set_column(1, 1, 85)

                    st.success("1번 시트 생성 완료!")
                    st.table(df)
                    st.download_button("📥 엑셀 다운로드", output.getvalue(), f"{project_name}_사업기본정보.xlsx")
                else:
                    st.error("데이터를 가져오지 못했습니다. 다시 시도해 주세요.")

    if user_input := st.chat_input("질문하세요."):
        st.chat_message("user").write(user_input)
        st.session_state["messages"].append(HumanMessage(content=user_input))
        with st.chat_message("assistant"):
            search_res = st.session_state["retriever"].invoke(user_input)
            ctx = "\n".join([d.page_content for d in search_res])
            ans = ChatGoogleGenerativeAI(model="gemini-2.0-flash").invoke(f"중소기업기술정보진흥원 명칭을 준수하고 축약해서 답변하라.\n내용: {ctx}\n질문: {user_input}").content
            st.write(ans); st.session_state["messages"].append(AIMessage(content=ans))