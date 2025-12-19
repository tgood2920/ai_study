import pandas as pd
import io
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI

def get_basic_info_json(docs):
    """PDF에서 1번 시트용 데이터를 JSON으로 추출"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    context = "\n\n".join([doc.page_content for doc in docs[:15]])
    
    prompt = f"""
    너는 제안서 작성 전문가야. 아래 내용을 분석해 JSON으로만 응답해.
    
    [규칙]:
    1. '중소기업기술정보진흥원' 명칭은 절대 줄이지 마라.
    2. 모든 내용은 최대한 요약해서 축약형으로 작성하라.
    3. 데이터가 없으면 빈 문자열("")로 표시하라.
    
    [항목]:
    공식사업명, 공고번호, 수요기관, 사업예산(VAT포함), 사업기간, 입찰방식, 낙찰자결정방법, 입찰참가자격, 담당자정보

    [내용]:
    {context}
    """
    try:
        res = llm.invoke(prompt).content
        match = re.search(r'\{.*\}', res, re.DOTALL)
        return json.loads(match.group(0)) if match else None
    except:
        return None

def create_basic_info_excel(data_dict, project_alias):
    """추출된 데이터를 바탕으로 이미지 양식과 유사한 엑셀 생성"""
    if not data_dict:
        return None
        
    df = pd.DataFrame(list(data_dict.items()), columns=['구분', '내용'])
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='1. 사업기본정보')
        
        workbook = writer.book
        worksheet = writer.sheets['1. 사업기본정보']
        
        # --- 서식 정의 ---
        header_fmt = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter',
            'bg_color': '#D7E4BC', 'border': 1, 'font_size': 11
        })
        cell_fmt = workbook.add_format({
            'valign': 'vcenter', 'border': 1, 'text_wrap': True, 'font_size': 10
        })
        
        # --- 표 꾸미기 ---
        worksheet.set_column(0, 0, 25) # 구분 열 너비
        worksheet.set_column(1, 1, 80) # 내용 열 너비
        
        # 헤더 쓰기
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_fmt)
            
        # 데이터 쓰기 (None 방지 및 문자열 변환)
        for row_num, (k, v) in enumerate(data_dict.items()):
            worksheet.write(row_num + 1, 0, str(k), cell_fmt)
            worksheet.write(row_num + 1, 1, str(v) if v else "", cell_fmt)
            
    return output.getvalue()