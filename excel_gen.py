import pandas as pd
import io
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI

def get_basic_info_json(docs):
    """RFP에서 섹션별 데이터를 통합 추출"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    context = "\n\n".join([doc.page_content for doc in docs[:15]])
    
    prompt = f"""
    너는 제안서 작성 전문가야. 아래 [내용]을 분석해 반드시 JSON으로만 응답해.
    
    [규칙]:
    1. 모든 내용은 최대한 요약해서 축약형으로 작성하라.
    3. 해당 데이터를 찾을 수 없다면 빈 문자열("")로 표시하라.

    [JSON 구조]:
    {{
        "basic": {{
            "공식사업명": "", "수요기관": "", "사업기간": "", "공고일": "", "담당부서": "", "사업비용": "", "입찰방식": ""
        }},
        "managers": [
            {{"소속": "", "성명": "", "연락처": "", "이메일": ""}}
        ],
        "issues": [
            {{"구분": "", "주요사항": "", "비고": ""}}
        ],
        "status": [
            {{"일자": "", "주요사항": "", "비고": ""}}
        ]
    }}

    [내용]:
    {context}
    """
    try:
        res = llm.invoke(prompt).content
        match = re.search(r'\{.*\}', res, re.DOTALL)
        return json.loads(match.group(0)) if match else None
    except:
        return None

def create_basic_info_excel(data, project_alias):
    """사용자가 제시한 append 방식의 섹션별 레이아웃 생성"""
    if not data:
        return None
        
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet('1. 사업기본정보')
        
        # --- 서식 정의 ---
        title_fmt = workbook.add_format({'bold': True}) # 섹션 제목
        head_fmt = workbook.add_format({'bold': True, 'align': 'center', 'bg_color': '#F2F2F2', 'border': 1, 'border-top': 2}) # 표 헤더
        cell_fmt = workbook.add_format({'border': 1, 'text_wrap': True, 'valign': 'vcenter'})
        
        curr_row = 0

        # 1. 사업기본정보 섹션
        worksheet.write(curr_row, 0, " # 사업기본정보", title_fmt)
        worksheet.merge_range(curr_row, 0, curr_row, 3, " # 사업기본정보", title_fmt)
        curr_row += 1
        # for k, v in data.get('basic', {}).items():
        #    worksheet.write(curr_row, 0, k, head_fmt)
        #    worksheet.write(curr_row, 1, str(v), cell_fmt)
        #    curr_row += 1
        basic_items = list(data.get('basic', {}).items())
        for i in range(0, len(basic_items), 2):
            # 첫 번째 항목
            worksheet.write(curr_row, 0, basic_items[i][0], head_fmt)
            worksheet.write(curr_row, 1, str(basic_items[i][1]), cell_fmt)
            # 두 번째 항목이 있다면
            if i + 1 < len(basic_items):
                worksheet.write(curr_row, 2, basic_items[i + 1][0], head_fmt)
                worksheet.write(curr_row, 3, str(basic_items[i + 1][1]), cell_fmt)
            curr_row += 1

        curr_row += 1 # 빈 행



        # 2. 사업담당자 섹션
        worksheet.merge_range(curr_row, 0, curr_row, 3, " # 사업담당자", title_fmt)
        curr_row += 1
        headers = ["소속", "성명", "연락처", "이메일"]
        for col, h in enumerate(headers):
            worksheet.write(curr_row, col, h, head_fmt)
        curr_row += 1
        for m in data.get('managers', []):
            worksheet.write(curr_row, 0, m.get("소속", ""), cell_fmt)
            worksheet.write(curr_row, 1, m.get("성명", ""), cell_fmt)
            worksheet.write(curr_row, 2, m.get("연락처", ""), cell_fmt)
            worksheet.write(curr_row, 3, m.get("이메일", ""), cell_fmt)
            curr_row += 1
        curr_row += 1

        # 3. 주요 이슈 섹션
        worksheet.merge_range(curr_row, 0, curr_row, 2, " # 주요 이슈(특이사항)", title_fmt)
        curr_row += 1
        headers = ["구분", "주요사항", "비고"]
        for col, h in enumerate(headers):
            worksheet.write(curr_row, col, h, head_fmt)
        curr_row += 1
        for i in data.get('issues', []):
            worksheet.write(curr_row, 0, i.get("구분", ""), cell_fmt)
            worksheet.write(curr_row, 1, i.get("주요사항", ""), cell_fmt)
            worksheet.write(curr_row, 2, i.get("비고", ""), cell_fmt)
            curr_row += 1
        curr_row += 1

        # 4. 사업진행현황 섹션
        worksheet.merge_range(curr_row, 0, curr_row, 2, " # 사업진행현황", title_fmt)
        curr_row += 1
        headers = ["일자", "주요사항", "비고"]
        for col, h in enumerate(headers):
            worksheet.write(curr_row, col, h, head_fmt)
        curr_row += 1
        for s in data.get('status', []):
            worksheet.write(curr_row, 0, s.get("일자", ""), cell_fmt)
            worksheet.write(curr_row, 1, s.get("주요사항", ""), cell_fmt)
            worksheet.write(curr_row, 2, s.get("비고", ""), cell_fmt)
            curr_row += 1

        # 열 너비 조정
        worksheet.set_column(0, 0, 20)
        worksheet.set_column(1, 1, 40)
        worksheet.set_column(2, 3, 25)

    return output.getvalue()