def write_sheet2(workbook, data, title_fmt, head_fmt, cell_fmt):
    ws = workbook.add_worksheet('2. 제안준비서류')
    curr_row = 0
    
    ws.merge_range(curr_row, 0, curr_row, 4, " # 제안준비서류 목록", title_fmt)
    curr_row += 1
    
    headers = ["순번", "서류명", "규격/수량", "제출방법", "비고"]
    for col, h in enumerate(headers):
        ws.write(curr_row, col, h, head_fmt)
    curr_row += 1
    
    prep_docs = data.get('prep_docs', [])
    for idx, doc in enumerate(prep_docs, start=1):
        ws.write(curr_row, 0, doc.get("순번", idx), cell_fmt)
        ws.write(curr_row, 1, doc.get("서류명", ""), cell_fmt)
        ws.write(curr_row, 2, doc.get("규격/수량", ""), cell_fmt)
        ws.write(curr_row, 3, doc.get("제출방법", ""), cell_fmt)
        ws.write(curr_row, 4, doc.get("비고", ""), cell_fmt)
        curr_row += 1
        
    ws.set_column(1, 1, 40)
    ws.set_column(2, 4, 15)