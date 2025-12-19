import xlsxwriter

def write_sheet1(workbook, data, title_fmt, head_fmt, cell_fmt):
    ws = workbook.add_worksheet('1. 사업기본정보')
    curr_row = 0
    # 데이터 구조 대응 (basic_info 내부의 basic 참조)
    b_data_full = data.get('basic_info', {})
    basic_data = b_data_full.get('basic', {})

    # --- 섹션 1: 사업기본정보 ---
    ws.merge_range(curr_row, 0, curr_row, 3, " # 사업기본정보", title_fmt)
    curr_row += 1
    
    # 1-1. 공식사업명 (독립행, 1~3열 병합)
    if "공식사업명" in basic_data:
        ws.write(curr_row, 0, "공식사업명", head_fmt)
        val_name = basic_data.pop("공식사업명") # 여기서 먼저 제거
        ws.merge_range(curr_row, 1, curr_row, 3, str(val_name), cell_fmt)
        curr_row += 1

    # 1-2. 나머지 항목들 (2열씩 배치)
    items = list(basic_data.items()) # pop 이후에 리스트 생성
    for i in range(0, len(items), 2):
        # 첫 번째 쌍 (A, B열)
        ws.write(curr_row, 0, items[i][0], head_fmt)
        ws.write(curr_row, 1, str(items[i][1]) if items[i][1] else "", cell_fmt)
        
        # 두 번째 쌍 (C, D열)
        if i + 1 < len(items):
            ws.write(curr_row, 2, items[i+1][0], head_fmt)
            ws.write(curr_row, 3, str(items[i+1][1]) if items[i+1][1] else "", cell_fmt)
        else:
            ws.write(curr_row, 2, "", head_fmt)
            ws.write(curr_row, 3, "", cell_fmt)
        curr_row += 1
    curr_row += 1

    # --- 섹션 2: 사업담당자 ---
    ws.merge_range(curr_row, 0, curr_row, 3, " # 사업담당자", title_fmt)
    curr_row += 1
    headers2 = ["소속", "성명", "연락처", "이메일"]
    for col, h in enumerate(headers2):
        ws.write(curr_row, col, h, head_fmt)
    curr_row += 1
    for m in b_data_full.get('managers', []):
        ws.write(curr_row, 0, m.get("소속", ""), cell_fmt)
        ws.write(curr_row, 1, m.get("성명", ""), cell_fmt)
        ws.write(curr_row, 2, m.get("연락처", ""), cell_fmt)
        ws.write(curr_row, 3, m.get("이메일", ""), cell_fmt)
        curr_row += 1
    curr_row += 1

    # --- 섹션 3: 주요 이슈(특이사항) ---
    ws.merge_range(curr_row, 0, curr_row, 3, " # 주요 이슈(특이사항)", title_fmt)
    curr_row += 1
    ws.write(curr_row, 0, "구분", head_fmt)
    ws.merge_range(curr_row, 1, curr_row, 2, "주요사항", head_fmt)
    ws.write(curr_row, 3, "비고", head_fmt)
    curr_row += 1
    for issue in b_data_full.get('issues', []):
        ws.write(curr_row, 0, issue.get("구분", ""), cell_fmt)
        ws.merge_range(curr_row, 1, curr_row, 2, str(issue.get("주요사항", "")), cell_fmt)
        ws.write(curr_row, 3, issue.get("비고", ""), cell_fmt)
        curr_row += 1
    curr_row += 1

    # --- 섹션 4: 사업진행현황 ---
    ws.merge_range(curr_row, 0, curr_row, 3, " # 사업진행현황", title_fmt)
    curr_row += 1
    ws.write(curr_row, 0, "일자", head_fmt)
    ws.merge_range(curr_row, 1, curr_row, 2, "주요사항", head_fmt)
    ws.write(curr_row, 3, "비고", head_fmt)
    curr_row += 1
    for status in b_data_full.get('status', []):
        ws.write(curr_row, 0, status.get("일자", ""), cell_fmt)
        ws.merge_range(curr_row, 1, curr_row, 2, str(status.get("주요사항", "")), cell_fmt)
        ws.write(curr_row, 3, status.get("비고", ""), cell_fmt)
        curr_row += 1

    ws.set_column(0, 0, 20)
    ws.set_column(1, 2, 35)
    ws.set_column(3, 3, 25)