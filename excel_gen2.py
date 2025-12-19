import xlsxwriter

def write_sheet2(workbook, data, title_fmt, head_fmt, cell_fmt):
    ws = workbook.add_worksheet('2. 제안준비서류')
    curr_row = 1
    
    p_data_raw = data.get('prep_docs', {})
    if isinstance(p_data_raw, list):
        p_data = p_data_raw[0] if p_data_raw else {}
    else:
        p_data = p_data_raw

    title_line_fmt = workbook.add_format({'bold': True, 'font_size': 16, 'font_color': '#002060', 'bottom': 2})
    label_fmt = workbook.add_format({'bold': True, 'align': 'center', 'bg_color': '#E7E6E6', 'border': 1, 'valign': 'vcenter'})
    cat_fmt = workbook.add_format({'bold': True, 'align': 'center', 'bg_color': '#D9E1F2', 'border': 1, 'valign': 'vcenter'})

    # 1. 타이틀: 입찰 서류 및 제안 제출 (B2~I2)
    ws.merge_range(curr_row, 1, curr_row, 3, "입찰 서류 및 제안 제출", title_line_fmt)
    curr_row += 1

    ws.write(curr_row, 1, "사업명", label_fmt)
    ws.merge_range(curr_row, 2, curr_row, 3, str(p_data.get("project_name", "")), cell_fmt)
    curr_row += 2 # 이미지처럼 한 칸 띄움

    ws.write(curr_row, 1, "제안서", label_fmt)
    ws.merge_range(curr_row, 1, "제안서", cat_fmt)
    ws.write(curr_row, 1, "제안서 제출 방식", label_fmt)
    ws.merge_range(curr_row, 2, curr_row, 3, str(p_data.get("sub_method", "")), cell_fmt)
    ws.write(curr_row, 1, "제출 부수", label_fmt)
    ws.merge_range(curr_row, 2, curr_row, 3, str(p_data.get("sub_copies", "")), cell_fmt)
    curr_row += 1
    # 하단 빈 줄 (이미지 양식 유지용)
    ws.merge_range(curr_row, 2, curr_row, 8, "", cell_fmt)
    curr_row += 3

    # --- 2섹션: 제출 서류 테이블 시작 ---
    ws.write(curr_row, 1, "제출 서류", workbook.add_format({'bold': True, 'font_size': 11}))
    curr_row += 1

    # 테이블 헤더 (남색 배경, 흰색 글자)
    tbl_head_fmt = workbook.add_format({
        'bold': True, 'align': 'center', 'bg_color': '#002060', 
        'font_color': 'white', 'border': 1, 'valign': 'vcenter'
    })
    
    ws.write(curr_row, 1, "구분", tbl_head_fmt)
    ws.write(curr_row, 1, "제출서류", tbl_head_fmt)
    ws.write(curr_row, 1, "확인사항", tbl_head_fmt)
    curr_row += 1

    # 테이블 데이터 (리스트 처리)
    doc_list = p_data.get("doc_list", [])
    if doc_list:
        for doc in doc_list:
            ws.write(curr_row, 1, doc.get("구분", ""), cell_fmt)
            ws.write(curr_row, 2 doc.get("제출서류", ""), cell_fmt)
            ws.write(curr_row, 3, doc.get("확인사항", ""), cell_fmt)
            curr_row += 1
    else:
        # 데이터 없을 시 빈 행 하나 생성
        ws.write(curr_row, 1, "", cell_fmt)
        ws.write(curr_row, 2, "", cell_fmt)
        ws.write(curr_row, 3, "", cell_fmt)

    ws.set_column(0, 0, 2)
    ws.set_column(1, 1, 15)
    ws.set_column(2, 2, 74)
    ws.set_column(3, 3, 35)
    
