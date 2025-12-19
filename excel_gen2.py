import xlsxwriter

def write_sheet2(workbook, data, title_fmt, head_fmt, cell_fmt):
    ws = workbook.add_worksheet('2. 제안준비서류')
    curr_row = 1
    
    # 1. 데이터 안전하게 가져오기 (리스트/딕셔너리 예외처리)
    p_data_raw = data.get('prep_docs', {})
    if isinstance(p_data_raw, list):
        p_data = p_data_raw[0] if p_data_raw else {}
    else:
        p_data = p_data_raw

    # 2. 서식 정의
    # 타이틀용 (파란 밑줄)
    title_line_fmt = workbook.add_format({'bold': True, 'font_size': 16, 'font_color': '#002060', 'bottom': 2})
    # 라벨 (회색 배경)
    label_fmt = workbook.add_format({'bold': True, 'align': 'center', 'bg_color': '#E7E6E6', 'border': 1, 'valign': 'vcenter'})
    # 카테고리 (연한 파랑 배경)
    cat_fmt = workbook.add_format({'bold': True, 'align': 'center', 'bg_color': '#D9E1F2', 'border': 1, 'valign': 'vcenter'})
    # 테이블 헤더 (남색 배경)
    tbl_head_fmt = workbook.add_format({'bold': True, 'align': 'center', 'bg_color': '#002060', 'font_color': 'white', 'border': 1, 'valign': 'vcenter'})

    # 3. 타이틀: 입찰 서류 및 제안 제출 (B~D열 병합)
    ws.merge_range(curr_row, 1, curr_row, 3, "입찰 서류 및 제안 제출", title_line_fmt)
    curr_row += 1

    # 4. 사업명 (B:라벨, C~D:내용 병합)
    ws.write(curr_row, 1, "사업명", label_fmt)
    ws.merge_range(curr_row, 2, curr_row, 3, str(p_data.get("project_name", "")), cell_fmt)
    curr_row += 2 

    # 5. 제안서 제출 정보
    # "제안서"라는 대분류를 B~D 전체 헤더로 처리하여 깔끔하게 정리
    ws.merge_range(curr_row, 1, curr_row, 3, "제안서 제출 정보", workbook.add_format({'bold': True, 'font_size': 12, 'bottom': 1}))
    curr_row += 1

    # 제안서 제출 방식
    ws.write(curr_row, 1, "제안서 제출 방식", label_fmt)
    ws.merge_range(curr_row, 2, curr_row, 3, str(p_data.get("sub_method", "")), cell_fmt)
    curr_row += 1
    
    # 제출 부수
    ws.write(curr_row, 1, "제출 부수", label_fmt)
    ws.merge_range(curr_row, 2, curr_row, 3, str(p_data.get("sub_copies", "")), cell_fmt)
    curr_row += 1
    
    # 빈 행 (공백)
    ws.merge_range(curr_row, 1, curr_row, 3, "", cell_fmt)
    curr_row += 2

    # --- 2섹션: 제출 서류 테이블 시작 ---
    ws.write(curr_row, 1, "제출 서류 목록", workbook.add_format({'bold': True, 'font_size': 11}))
    curr_row += 1

    # 테이블 헤더 (B, C, D열 각각 사용)
    ws.write(curr_row, 1, "구분", tbl_head_fmt)
    ws.write(curr_row, 2, "제출서류", tbl_head_fmt)
    ws.write(curr_row, 3, "확인사항", tbl_head_fmt)
    curr_row += 1

    # 테이블 데이터 (리스트 처리)
    doc_list = p_data.get("doc_list", [])
    if doc_list:
        for doc in doc_list:
            ws.write(curr_row, 1, doc.get("구분", ""), cell_fmt)
            # [수정] 쉼표(,) 추가됨
            ws.write(curr_row, 2, doc.get("제출서류", ""), cell_fmt)
            ws.write(curr_row, 3, doc.get("확인사항", ""), cell_fmt)
            curr_row += 1
    else:
        # 데이터 없을 시 빈 행 생성
        ws.write(curr_row, 1, "", cell_fmt)
        ws.write(curr_row, 2, "", cell_fmt)
        ws.write(curr_row, 3, "", cell_fmt)

    # 6. 열 너비 설정 (3열 구조: B, C, D)
    ws.set_column(0, 0, 2)   # A열 (여백)
    ws.set_column(1, 1, 20)  # B열 (구분/라벨)
    ws.set_column(2, 2, 60)  # C열 (제출서류 - 가장 넓게)
    ws.set_column(3, 3, 30)  # D열 (확인사항)