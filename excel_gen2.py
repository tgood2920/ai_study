def write_sheet2(workbook, data, title_fmt, head_fmt, cell_fmt):
    ws = workbook.add_worksheet('2. 제안준비서류')
    curr_row = 1
    p_data = data.get('prep_docs', {})

    # 스타일 정의
    label_fmt = workbook.add_format({'bold': True, 'align': 'center', 'bg_color': '#E7E6E6', 'border': 1})
    cat_fmt = workbook.add_format({'bold': True, 'align': 'center', 'bg_color': '#D9E1F2', 'border': 1, 'valign': 'vcenter'})
    title_line_fmt = workbook.add_format({'bold': True, 'font_size': 14, 'font_color': '#002060', 'bottom': 2})

    # 타이틀: 입찰 서류 및 제안 제출
    ws.merge_range(curr_row, 1, curr_row, 8, "입찰 서류 및 제안 제출", title_line_fmt)
    curr_row += 1

    # 1섹션: 사업명
    ws.write(curr_row, 1, "사업명", label_fmt)
    ws.merge_range(curr_row, 2, curr_row, 8, p_data.get("project_name", ""), cell_fmt)
    curr_row += 2 # 한 칸 띄움

    # 1섹션: 제안서 상세 (제출 방식, 부수)
    ws.merge_range(curr_row, 1, curr_row + 2, 1, "제안서", cat_fmt)
    
    ws.write(curr_row, 2, "제안서 제출 방식", label_fmt)
    ws.merge_range(curr_row, 3, curr_row, 6, p_data.get("sub_method", ""), cell_fmt)
    ws.merge_range(curr_row, 7, curr_row, 8, "", cell_fmt) # 빈 칸
    curr_row += 1

    ws.write(curr_row, 2, "제출 부수", label_fmt)
    ws.merge_range(curr_row, 3, curr_row, 6, p_data.get("sub_copies", ""), cell_fmt)
    ws.merge_range(curr_row, 7, curr_row, 8, "", cell_fmt)
    curr_row += 1
    
    # 빈 행 (부수 아래 여백)
    ws.merge_range(curr_row, 3, curr_row, 8, "", cell_fmt)
    curr_row += 3

    # 2섹션: 제출 서류 테이블 제목
    ws.write(curr_row, 1, "제출 서류", workbook.add_format({'bold': True}))
    curr_row += 1

    # 테이블 헤더 (구분, 제출서류, 확인사항)
    tbl_head_fmt = workbook.add_format({'bold': True, 'align': 'center', 'bg_color': '#002060', 'font_color': 'white', 'border': 1})
    ws.write(curr_row, 1, "구분", tbl_head_fmt)
    ws.merge_range(curr_row, 2, curr_row, 6, "제출서류", tbl_head_fmt)
    ws.merge_range(curr_row, 7, curr_row, 8, "확인사항", tbl_head_fmt)
    curr_row += 1

    # 테이블 데이터
    for doc in p_data.get("doc_list", []):
        ws.write(curr_row, 1, doc.get("구분", ""), cell_fmt)
        ws.merge_range(curr_row, 2, curr_row, 6, doc.get("제출서류", ""), cell_fmt)
        ws.merge_range(curr_row, 7, curr_row, 8, doc.get("확인사항", ""), cell_fmt)
        curr_row += 1

    # 열 너비 조정
    ws.set_column(1, 1, 15) # 구분
    ws.set_column(2, 6, 12) # 서류명 내용들
    ws.set_column(7, 8, 20) # 확인사항