import xlsxwriter

def write_sheet1(workbook, data, title_fmt, head_fmt, cell_fmt):
    ws = workbook.add_worksheet('1. 사업기본정보')
    curr_row = 0
    b_data = data.get('basic_info', {})

    # --- 섹션 1: 사업기본정보 (2열 배치) ---
    ws.merge_range(curr_row, 0, curr_row, 3, " # 사업기본정보", title_fmt)
    curr_row += 1
    items = list(b_data.get('basic', {}).items())
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

    # --- 섹션 2: 사업담당자 (표 형식) ---
    ws.merge_range(curr_row, 0, curr_row, 3, " # 사업담당자", title_fmt)
    curr_row += 1
    headers2 = ["소속", "성명", "연락처", "이메일"]
    for col, h in enumerate(headers2): ws.write(curr_row, col, h, head_fmt)
    curr_row += 1
    for m in b_data.get('managers', []):
        for col, key in enumerate(headers2): ws.write(curr_row, col, m.get(key, ""), cell_fmt)
        curr_row += 1
    curr_row += 1

    # --- 섹션 3: 주요 이슈(특이사항) (표 형식) ---
    ws.merge_range(curr_row, 0, curr_row, 3, " # 주요 이슈(특이사항)", title_fmt)
    curr_row += 1
    headers3 = ["구분", "주요사항", "비고", ""] # 4열 대응 위해 빈칸 추가
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

    # --- 섹션 4: 사업진행현황 (표 형식) ---
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

    ws.set_column(0, 3, 20) # 전체 열 너비 조정