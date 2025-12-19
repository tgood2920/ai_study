def write_sheet1(workbook, data, title_fmt, head_fmt, cell_fmt):
    ws = workbook.add_worksheet('1. 사업기본정보')
    curr_row = 0
    b_data = data.get('basic_info', {})

    # 섹션 1: 사업기본정보 (2열 배치)
    ws.merge_range(curr_row, 0, curr_row, 3, " # 사업기본정보", title_fmt)
    curr_row += 1
    items = list(b_data.get('basic', {}).items())
    if "공식사업명" in basic_data:
            worksheet.write(curr_row, 0, "공식사업명", head_fmt)
            worksheet.merge_range(curr_row, 1, curr_row, 3, str(basic_data["공식사업명"]), cell_fmt)
            curr_row += 1
            basic_data.pop("공식사업명")
    for i in range(0, len(items), 2):
        ws.write(curr_row, 0, items[i][0], head_fmt)
        ws.write(curr_row, 1, str(items[i][1]) if items[i][1] else "", cell_fmt)
        if i + 1 < len(items):
            ws.write(curr_row, 2, items[i+1][0], head_fmt)
            ws.write(curr_row, 3, str(items[i+1][1]) if items[i+1][1] else "", cell_fmt)
        else:
            ws.write(curr_row, 2, "", head_fmt); ws.write(curr_row, 3, "", cell_fmt)
        curr_row += 1
    curr_row += 1

    # 섹션 2: 사업담당자
    ws.merge_range(curr_row, 0, curr_row, 3, " # 사업담당자", title_fmt)
    curr_row += 1
    headers2 = ["소속", "성명", "연락처", "이메일"]
    for col, h in enumerate(headers2): ws.write(curr_row, col, h, head_fmt)
    curr_row += 1
    for m in b_data.get('managers', []):
        for col, key in enumerate(headers2): ws.write(curr_row, col, m.get(key, ""), cell_fmt)
        curr_row += 1
    curr_row += 1

    # 섹션 3: 주요 이슈(특이사항) - 1,2열 병합
    ws.merge_range(curr_row, 0, curr_row, 3, " # 주요 이슈(특이사항)", title_fmt)
    curr_row += 1
    ws.write(curr_row, 0, "구분", head_fmt)
    ws.merge_range(curr_row, 1, curr_row, 2, "주요사항", head_fmt)
    ws.write(curr_row, 3, "비고", head_fmt)
    curr_row += 1
    for issue in b_data.get('issues', []):
        ws.write(curr_row, 0, issue.get("구분", ""), cell_fmt)
        ws.merge_range(curr_row, 1, curr_row, 2, issue.get("주요사항", ""), cell_fmt)
        ws.write(curr_row, 3, issue.get("비고", ""), cell_fmt)
        curr_row += 1
    curr_row += 1

    # 섹션 4: 사업진행현황 - 1,2열 병합
    ws.merge_range(curr_row, 0, curr_row, 3, " # 사업진행현황", title_fmt)
    curr_row += 1
    ws.write(curr_row, 0, "일자", head_fmt)
    ws.merge_range(curr_row, 1, curr_row, 2, "주요사항", head_fmt)
    ws.write(curr_row, 3, "비고", head_fmt)
    curr_row += 1
    for status in b_data.get('status', []):
        ws.write(curr_row, 0, status.get("일자", ""), cell_fmt)
        ws.merge_range(curr_row, 1, curr_row, 2, status.get("주요사항", ""), cell_fmt)
        ws.write(curr_row, 3, status.get("비고", ""), cell_fmt)
        curr_row += 1

    ws.set_column(0, 0, 15); ws.set_column(1, 2, 30); ws.set_column(3, 3, 25)




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

        worksheet.write(curr_row, 0, "구분", head_fmt)
        worksheet.merge_range(curr_row, 1, curr_row, 2, "주요사항", head_fmt)
        worksheet.write(curr_row, 3, "비고", head_fmt)
        curr_row += 1

        for i in data.get('issues', []):
            worksheet.write(curr_row, 0, i.get("구분", ""), cell_fmt)
            val_issue = i.get("주요사항", "")
            worksheet.merge_range(curr_row, 1, curr_row, 2, str(val_issue), cell_fmt)
            worksheet.write(curr_row, 3, i.get("비고", ""), cell_fmt)
            curr_row += 1
        curr_row += 1

        # 4. 사업진행현황 섹션
        worksheet.merge_range(curr_row, 0, curr_row, 3, " # 사업진행현황", title_fmt) # 0~3까지 제목 병합
        curr_row += 1

        worksheet.write(curr_row, 0, "일자", head_fmt)
        worksheet.merge_range(curr_row, 1, curr_row, 2, "주요사항", head_fmt)
        worksheet.write(curr_row, 3, "비고", head_fmt)
        curr_row += 1

        for s in data.get('status', []):
            worksheet.write(curr_row, 0, s.get("일자", ""), cell_fmt)
            val_status = s.get("주요사항", "")
            worksheet.merge_range(curr_row, 1, curr_row, 2, str(val_status), cell_fmt)
            worksheet.write(curr_row, 3, s.get("비고", ""), cell_fmt)
            curr_row += 1

        worksheet.set_column(0, 0, 20)
        worksheet.set_column(1, 1, 40)
        worksheet.set_column(2, 3, 25)

    return output.getvalue()