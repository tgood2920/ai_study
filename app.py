import streamlit as st
import pandas as pd
import io
import json
import re
from excel_gen1 import write_sheet1
from excel_gen2 import write_sheet2
# (상단 import 및 PDF 처리 함수는 이전과 동일하게 유지)

def get_integrated_data(docs):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    context = "\n\n".join([doc.page_content for doc in docs[:15]]) + \
              "\n\n" + "\n\n".join([doc.page_content for doc in docs[-10:]])
    
    prompt = f"""
    RFP를 분석해 JSON으로 응답해. 모든 내용은 최대한 짧게 축약해.
    
    구조: {{
        "basic_info": {{
            "basic": {{ "공식사업명":"", "공고번호":"", "수요기관":"", "사업예산":"", "사업기간":"", "입찰방식":"" }},
            "managers": [ {{ "소속":"", "성명":"", "연락처":"", "이메일":"" }} ],
            "issues": [ {{ "구분":"", "주요사항":"", "비고":"" }} ],
            "status": [ {{ "일자":"", "주요사항":"", "비고":"" }} ]
        }},
        "prep_docs": [ {{ "순번":1, "서류명":"", "규격/수량":"", "제출방법":"", "비고":"" }} ]
    }}
    내용: {context}
    """
    res = llm.invoke(prompt).content
    match = re.search(r'\{.*\}', res, re.DOTALL)
    return json.loads(match.group(0)) if match else None

# --- UI 메인 로직 ---
if st.session_state["analysis_done"]:
    if st.button("📊 통합 엑셀 생성 (1, 2번 시트)"):
        with st.spinner("이미지 양식에 맞춰 시트 구성 중..."):
            data = get_integrated_data(st.session_state["docs"])
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                wb = writer.book
                # 공통 서식 (개발자님이 설정한 스타일)
                t_fmt = wb.add_format({'bold': True, 'font_size': 12})
                h_fmt = wb.add_format({'bold': True, 'align': 'center', 'bg_color': '#F2F2F2', 'border': 1})
                c_fmt = wb.add_format({'border': 1, 'text_wrap': True, 'valign': 'vcenter'})
                
                # 시트별 모듈 호출
                write_sheet1(wb, data, t_fmt, h_fmt, c_fmt)
                write_sheet2(wb, data, t_fmt, h_fmt, c_fmt)
            
            st.download_button("📥 다운로드", output.getvalue(), f"{project_alias}_제안요약.xlsx")