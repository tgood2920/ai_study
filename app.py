import streamlit as st
import os
import time
import pandas as pd
import io
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

# 1. 환경 설정 및 초기화
load_dotenv()
st.set_page_config(page_title="RFP 입찰 분석 & 스토리보드 생성기", page_icon="📑", layout="wide")

# [중요] 세션 상태 초기화 (KeyError 방지)
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

st.title("📑 RFP 분석 & 제안 스토리보드 생성기")
st.markdown("""
입찰 공고(RFP)를 분석하여 **핵심 요약**, **독소 조항 체크**, 그리고 **제안서 목차(Excel)**까지 한 번에 생성합니다.
""")

# 2. PDF 처리 및 벡터 DB 생성 함수
@st.cache_resource
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    return vectorstore.as_retriever(), docs

# 3. RFP 분석 함수 (요약 및 리스크)
def analyze_rfp(docs, project_name):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    # 앞부분(개요)과 뒷부분(평가) 등 주요 부분 참조
    # 문서가 너무 길 경우를 대비해 앞 10페이지와 뒤 5페이지를 조합
    ref_docs = docs[:10] + docs[-5:]
    context_text = "\n\n".join([doc.page_content for doc in ref_docs])

    prompt = f"""
    너는 20년 차 공공사업 제안 PM이야.
    [제안요청서(RFP)]를 분석해서 핵심을 정리해줘.
    
    [사업명]: {project_name}
    [RFP 내용 일부]: {context_text}
    
    [출력 양식]:
    ## 1. 🎯 사업 개요
    * **사업 목적**: (한 줄 요약)
    * **예산 및 기간**: (예산 / 기간)
    * **주요 과업**: (핵심 요구사항 3가지)

    ## 2. ⚠️ 리스크 및 제약사항 (독소조항)
    * **입찰 자격**: (제한사항)
    * **인력 요건**: (PM등급, 상주 여부 등)
    * **페널티/제약**: (지체상금률, 기술이전 등 특이사항)
    
    ## 3. 💡 제안 전략 가이드
    * 이 사업을 수주하기 위해 강조해야 할 차별화 포인트 3가지.
    """
    response = llm.invoke(prompt)
    return response.content

# 4. [New] 스토리보드(엑셀) 데이터 생성 함수
def generate_storyboard_data(docs, project_name):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
    
    # 목차 생성을 위해 문서 전반적인 맥락 필요
    ref_docs = docs[:15] + docs[-5:]
    context_text = "\n\n".join([doc.page_content for doc in ref_docs])
    
    prompt = f"""
    너는 제안서 작성을 총괄하는 메인 기획자야.
    RFP 내용을 바탕으로 **상세 제안 목차(스토리보드)**를 엑셀로 만들려고 해.
    일반적인 공공 SI 제안서 표준 목차(전략, 개요, 기술, 관리, 지원)를 따르되, 
    RFP의 요구사항과 평가항목을 적절한 목차에 배치해줘.

    [지시사항]:
    1. **반드시 아래 CSV 형식(파이프라인 | 구분)으로만 출력해.** (사족 붙이지 마)
    2. '핵심작성내용'에는 해당 목차에 들어가야 할 차별화 전략이나 필수 내용을 적어.
    3. '예상페이지'는 전체 200페이지 기준으로 중요도에 따라 배분해.
    
    [CSV 출력 포맷]:
    대목차|중목차|소목차|핵심작성내용(Key Message)|관련요구사항ID|예상페이지
    I. 제안개요|1. 제안배경|1.1 추진배경 및 목적|사업 이해도 및 기대효과 강조|REQ-001|2
    I. 제안개요|2. 사업범위|2.1 목표시스템 구성|To-Be 모델 아키텍처 제시|REQ-002|3
    ... (계속 작성) ...
    """
    
    response = llm.invoke(prompt)
    return response.content

# --- UI 구성 ---

with st.sidebar:
    st.header("📂 프로젝트 설정")
    project_name = st.text_input("사업명", value="차세대 정보시스템 구축 사업")
    uploaded_file = st.file_uploader("RFP(PDF) 업로드", type=["pdf"])
    
    st.divider()
    st.info("💡 PDF를 업로드하면 자동으로 분석이 시작됩니다.")

# 메인 로직
if uploaded_file is not None:
    # 1. 파일 저장 및 세션 관리
    temp_pdf_path = f"temp_{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 파일 변경 감지 및 초기화
    if "last_uploaded_file" not in st.session_state or st.session_state["last_uploaded_file"] != uploaded_file.name:
        st.session_state["last_uploaded_file"] = uploaded_file.name
        st.session_state["messages"] = []
        st.session_state["analysis_done"] = False
        if "retriever" in st.session_state:
            del st.session_state["retriever"]

    try:
        # 2. PDF 처리 (한 번만 실행)
        if "retriever" not in st.session_state:
            with st.spinner(f"🔍 '{project_name}' RFP 분석 중... (잠시만 기다려주세요)"):
                retriever, docs = process_pdf(temp_pdf_path)
                st.session_state["retriever"] = retriever
                st.session_state["docs"] = docs # 엑셀 생성을 위해 원본 저장
                
                # 3. 기본 분석 수행
                analysis_result = analyze_rfp(docs, project_name)
                
                # 결과 메시지 저장
                welcome_msg = AIMessage(content=f"**[{project_name}]** 분석 완료! 🚀\n\n{analysis_result}")
                st.session_state["messages"].append(welcome_msg)
                st.session_state["analysis_done"] = True

        # --- 화면 표시 영역 ---
        
        # (1) 채팅창 (분석 결과 및 대화)
        for msg in st.session_state["messages"]:
            if isinstance(msg, HumanMessage):
                st.chat_message("user").write(msg.content)
            elif isinstance(msg, AIMessage):
                st.chat_message("assistant", avatar="👨‍💼").write(msg.content)

        # (2) [New] 스토리보드 생성 버튼 (분석 완료 시에만 표시)
        if st.session_state["analysis_done"]:
            st.divider()
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("### 📊 제안 스토리보드(Excel) 만들기")
                st.markdown("RFP 내용을 기반으로 **상세 목차, 페이지 계획, 핵심 전략**이 담긴 엑셀 파일을 생성합니다.")
            
            with col2:
                generate_btn = st.button("스토리보드 생성 ✨")
            
            if generate_btn:
                with st.spinner("AI가 제안 전략을 짜고 엑셀을 만드는 중입니다..."):
                    docs = st.session_state["docs"]
                    csv_data = generate_storyboard_data(docs, project_name)
                    
                    # 데이터 전처리 (CSV 파싱)
                    try:
                        valid_lines = [line for line in csv_data.split('\n') if '|' in line]
                        clean_csv = "\n".join(valid_lines)
                        
                        df = pd.read_csv(io.StringIO(clean_csv), sep="|")
                        
                        # 엑셀 변환 (메모리 상에서 처리)
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df.to_excel(writer, index=False, sheet_name='제안목차_v1.0')
                            
                            # (선택) 엑셀 스타일링
                            workbook = writer.book
                            worksheet = writer.sheets['제안목차_v1.0']
                            header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
                            for col_num, value in enumerate(df.columns.values):
                                worksheet.write(0, col_num, value, header_fmt)
                                worksheet.set_column(col_num, col_num, 20) # 너비 조절

                        excel_data = output.getvalue()
                        
                        st.success("생성 완료! 아래 버튼을 눌러 다운로드하세요.")
                        st.dataframe(df) # 미리보기
                        
                        st.download_button(
                            label="📥 엑셀 파일 다운로드 (.xlsx)",
                            data=excel_data,
                            file_name=f"{project_name}_제안스토리보드.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        st.error("엑셀 생성 중 오류가 발생했습니다. AI 응답을 확인해주세요.")
                        with st.expander("AI 원본 응답 보기"):
                            st.write(csv_data)

        # (3) 사용자 추가 질문 입력
        if user_input := st.chat_input("RFP에 대해 궁금한 점을 물어보세요 (예: 평가 배점이 어떻게 돼?)"):
            st.chat_message("user").write(user_input)
            st.session_state["messages"].append(HumanMessage(content=user_input))

            with st.chat_message("assistant", avatar="👨‍💼"):
                message_placeholder = st.empty()
                full_response = ""
                
                with st.spinner("RFP 확인 중..."):
                    retriever = st.session_state["retriever"]
                    retrieved_docs = retriever.invoke(user_input)
                    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # 채팅용 프롬프트
                prompt_template = ChatPromptTemplate.from_template(f"""
                너는 제안 PM이야. RFP 내용을 근거로 답변해.
                [참고 자료]: {{context}}
                [질문]: {{input}}
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
            
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")

else:
    # 파일 없을 때 안내
    st.info("👈 왼쪽 사이드바에서 PDF 제안요청서를 업로드해주세요.")