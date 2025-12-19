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
            "공식사업명": "", "수요기관": "", "공고일": "", "사업비용": "", "사업기간": "",  "담당부서": "",  "입찰방식": ""
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
        title_fmt = workbook.add_format({'bold': True, 'font_size': 12, 'bottom' : 2})
        head_fmt = workbook.add_format({'bold': True, 'align': 'center', 'bg_color': '#F2F2F2', 'border': 1}) 
        cell_fmt = workbook.add_format({'border': 1, 'text_wrap': True, 'valign': 'vcenter'})
        
        curr_row = 0

        # 1. 사업기본정보 섹션 시작
        worksheet.merge_range(curr_row, 0, curr_row, 3, " # 사업기본정보", title_fmt)
        curr_row += 1
        
        basic_data = data.get('basic', {})

        if "공식사업명" in basic_data:
            worksheet.write(curr_row, 0, "공식사업명", head_fmt)
            worksheet.merge_range(curr_row, 1, curr_row, 3, str(basic_data["공식사업명"]), cell_fmt)
            curr_row += 1
            basic_data.pop("공식사업명")

        items = list(basic_data.items())
        for i in range(0, len(items), 2):
            worksheet.write(curr_row, 0, items[i][0], head_fmt)
            worksheet.write(curr_row, 1, str(items[i][1]) if items[i][1] else "", cell_fmt)

            if i + 1 < len(items):
                worksheet.write(curr_row, 2, items[i+1][0], head_fmt)
                worksheet.write(curr_row, 3, str(items[i+1][1]) if items[i+1][1] else "", cell_fmt)
            else:
                worksheet.write(curr_row, 2, "", head_fmt)
                worksheet.write(curr_row, 3, "", cell_fmt)
            curr_row += 1

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
        worksheet.merge_range(curr_row, 0, curr_row, 3, " # 주요 이슈(특이사항)", title_fmt)
        curr_row += 1

        # 헤더 병합 (구분:0, 주요사항:1~2합병, 비고:3)
        worksheet.write(curr_row, 0, "구분", head_fmt)
        worksheet.merge_range(curr_row, 1, curr_row, 2, "주요사항", head_fmt)
        worksheet.write(curr_row, 3, "비고", head_fmt)
        curr_row += 1

        for i in data.get('issues', []):
            worksheet.write(curr_row, 0, i.get("구분", ""), cell_fmt)
            # 1~2열 병합하여 주요사항 기입
            val_issue = i.get("주요사항", "")
            worksheet.merge_range(curr_row, 1, curr_row, 2, str(val_issue), cell_fmt)
            # 3열 비고 기입
            worksheet.write(curr_row, 3, i.get("비고", ""), cell_fmt)
            curr_row += 1
        curr_row += 1

        # 4. 사업진행현황 섹션
        worksheet.merge_range(curr_row, 0, curr_row, 3, " # 사업진행현황", title_fmt) # 0~3까지 제목 병합
        curr_row += 1

        # 헤더 병합 (일자:0, 주요사항:1~2합병, 비고:3)
        worksheet.write(curr_row, 0, "일자", head_fmt)
        worksheet.merge_range(curr_row, 1, curr_row, 2, "주요사항", head_fmt)
        worksheet.write(curr_row, 3, "비고", head_fmt)
        curr_row += 1

        for s in data.get('status', []):
            worksheet.write(curr_row, 0, s.get("일자", ""), cell_fmt)
            # 1~2열 병합하여 주요사항 기입
            val_status = s.get("주요사항", "")
            worksheet.merge_range(curr_row, 1, curr_row, 2, str(val_status), cell_fmt)
            # 3열 비고 기입
            worksheet.write(curr_row, 3, s.get("비고", ""), cell_fmt)
            curr_row += 1

        # 열 너비 조정
        worksheet.set_column(0, 0, 20)
        worksheet.set_column(1, 1, 40)
        worksheet.set_column(2, 3, 25)

    return output.getvalue()