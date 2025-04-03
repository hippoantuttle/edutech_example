import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import random

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 환경
# plt.rcParams['font.family'] = 'AppleGothic'  # Mac 환경
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 해결

# 스트림릿 앱 설정
st.title("원(Circle)의 개념 이해하기")
st.write("이 애플리케이션은 원의 정의와 특성을 시각적으로 보여줍니다.")

# 탭 생성
tab1, tab2 = st.tabs(["일정한 반지름 (원)", "랜덤 반지름 (원이 아닌 도형)"])

with tab1:
    # 사이드바에 원 매개변수 입력 위젯 생성
    st.sidebar.header("원의 매개변수 설정")
    center_x = st.sidebar.number_input("중심 X좌표", value=0.0, step=0.5, key="cx1")
    center_y = st.sidebar.number_input("중심 Y좌표", value=0.0, step=0.5, key="cy1")
    radius = st.sidebar.slider("반지름", min_value=0.1, max_value=10.0, value=5.0, step=0.1, key="r1")
    num_points = st.sidebar.slider("점의 갯수", min_value=3, max_value=100, value=20, step=1, key="np1")

    # 표시 옵션
    st.sidebar.header("표시 옵션")
    show_center = st.sidebar.checkbox("중심점 표시", value=True, key="sc1")
    show_radii = st.sidebar.checkbox("반지름 표시", value=True, key="sr1")
    show_complete_circle = st.sidebar.checkbox("완전한 원 표시", value=True, key="scc1")
    show_point_labels = st.sidebar.checkbox("점 라벨 표시", value=False, key="spl1")
    show_triangles = st.sidebar.checkbox("삼각형 표시", value=True, key="st1")
    show_polygon = st.sidebar.checkbox("다각형 표시", value=True, key="sp1")

    # 원 위의 점들 생성 함수
    def generate_circle_points(center_x, center_y, radius, num_points):
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append((x, y, angle, radius))  # 각도와 반지름 정보도 저장
        return points

    # 삼각형 넓이 계산 함수
    def calculate_triangle_area(center_x, center_y, x1, y1, x2, y2):
        # 벡터의 외적을 이용한 삼각형 넓이 계산
        v1_x, v1_y = x1 - center_x, y1 - center_y
        v2_x, v2_y = x2 - center_x, y2 - center_y
        cross_product = abs(v1_x * v2_y - v1_y * v2_x)
        return cross_product / 2

    # 점 생성
    points = generate_circle_points(center_x, center_y, radius, num_points)
    points_x = [p[0] for p in points]
    points_y = [p[1] for p in points]
    points_angle = [p[2] for p in points]
    points_radius = [p[3] for p in points]

    # 삼각형 넓이 계산
    triangle_areas = []
    for i in range(num_points):
        next_i = (i + 1) % num_points
        area = calculate_triangle_area(
            center_x, center_y, 
            points_x[i], points_y[i], 
            points_x[next_i], points_y[next_i]
        )
        triangle_areas.append(area)

    total_triangle_area = sum(triangle_areas)
    actual_circle_area = math.pi * radius ** 2

    # 다각형 둘레 계산
    # 근사 다각형의 한 변의 길이
    side_length = 2 * radius * math.sin(math.pi / num_points)
    polygon_perimeter = num_points * side_length
    actual_circle_perimeter = 2 * math.pi * radius

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(8, 8))

    # 삼각형 그리기 (선택된 경우)
    if show_triangles:
        for i in range(num_points):
            next_i = (i + 1) % num_points
            triangle = plt.Polygon([
                (center_x, center_y), 
                (points_x[i], points_y[i]), 
                (points_x[next_i], points_y[next_i])
            ], fill=True, alpha=0.1, color='orange')
            ax.add_patch(triangle)

    # 다각형 그리기 (선택된 경우)
    if show_polygon:
        polygon_points = [(x, y) for x, y in zip(points_x, points_y)]
        polygon = plt.Polygon(polygon_points, fill=False, edgecolor='purple', linewidth=2)
        ax.add_patch(polygon)

    # 완전한 원 그리기 (선택된 경우)
    if show_complete_circle:
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = center_x + radius * np.cos(theta)
        circle_y = center_y + radius * np.sin(theta)
        ax.plot(circle_x, circle_y, 'b-', alpha=0.3, label='원')

    # 점 그리기
    ax.scatter(points_x, points_y, color='red', s=50, zorder=2, label='원 위의 점들')

    # 중심 그리기 (선택된 경우)
    if show_center:
        ax.scatter([center_x], [center_y], color='green', s=100, zorder=3, label='중심')

    # 반지름 선 그리기 (선택된 경우)
    if show_radii:
        for i in range(num_points):
            ax.plot([center_x, points_x[i]], [center_y, points_y[i]], 'g-', alpha=0.5)

    # 점 라벨 추가 (선택된 경우)
    if show_point_labels:
        for i in range(num_points):
            angle_deg = math.degrees(points_angle[i])
            ax.annotate(f'P{i+1} ({angle_deg:.1f}°)', 
                      (points_x[i], points_y[i]), 
                      xytext=(points_x[i] + 0.2, points_y[i] + 0.2))

    # 그래프 설정
    ax.set_aspect('equal')
    margin = 1.2
    ax.set_xlim(center_x - radius * margin, center_x + radius * margin)
    ax.set_ylim(center_y - radius * margin, center_y + radius * margin)
    ax.grid(True)
    ax.set_xlabel('X-축')
    ax.set_ylabel('Y-축')
    ax.set_title(f'중심 ({center_x}, {center_y}), 반지름 {radius}인 원')
    ax.legend()

    # 그래프 표시
    st.pyplot(fig)

    # 설명 텍스트 추가
    st.markdown("""
    ## 원의 정의
    원(Circle)은 한 고정된 점(중심)으로부터 같은 거리(반지름)에 있는 모든 점들의 집합입니다.

    ## 수학적 표현
    중심이 (a, b)이고 반지름이 r인 원의 방정식: (x - a)² + (y - b)² = r²

    ## 주요 개념
    - **중심(Center)**: 원 위의 모든 점으로부터 같은 거리에 있는 고정된 점입니다.
    - **반지름(Radius)**: 중심에서 원 위의 임의의 점까지의 거리입니다.
    - **원주(Circumference)**: 원의 둘레입니다. 공식: 2πr
    - **원의 넓이(Area)**: 원이 차지하는 공간입니다. 공식: πr²
    """)

    # 삼각형 넓이와 원의 넓이 비교
    st.subheader("삼각형으로 근사한 넓이와 원의 넓이 비교")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**삼각형들의 총 넓이:** {total_triangle_area:.3f}")
        st.write(f"**실제 원의 넓이:** {actual_circle_area:.3f}")

    with col2:
        area_error = abs(total_triangle_area - actual_circle_area)
        area_error_percent = (area_error / actual_circle_area) * 100
        st.write(f"**오차:** {area_error:.3f}")
        st.write(f"**오차율:** {area_error_percent:.3f}%")

    st.write(f"점이 {num_points}개 일 때, 삼각형들의 넓이 합은 원의 넓이에 근사합니다.")
    st.write("점의 수가 많아질수록 더 정확한 근사치를 얻을 수 있습니다.")

    # 다각형 둘레와 원의 둘레 비교
    st.subheader("다각형 둘레와 원의 둘레 비교")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**{num_points}각형의 둘레:** {polygon_perimeter:.3f}")
        st.write(f"**실제 원의 둘레:** {actual_circle_perimeter:.3f}")

    with col2:
        perimeter_error = abs(polygon_perimeter - actual_circle_perimeter)
        perimeter_error_percent = (perimeter_error / actual_circle_perimeter) * 100
        st.write(f"**오차:** {perimeter_error:.3f}")
        st.write(f"**오차율:** {perimeter_error_percent:.3f}%")

    st.write(f"점이 {num_points}개(즉, {num_points}각형)일 때, 다각형의 둘레는 원의 둘레에 근사합니다.")
    st.write("점의 수가 많아질수록 더 정확한 근사치를 얻을 수 있습니다.")

    # 원 위의 점 정보 간략화 (토글 가능)
    with st.expander("원 위의 점 정보 (클릭하여 펼치기)"):
        # 원 위의 점들의 좌표 표시
        st.subheader("원 위의 점들의 좌표")
        point_data = []
        for i, point in enumerate(points):
            angle_deg = math.degrees(point[2])
            point_data.append({
                "점": f"P{i+1}",
                "X": round(point[0], 3),
                "Y": round(point[1], 3),
                "각도 (도)": f"{angle_deg:.2f}°"
            })
        st.table(point_data)

        # 원 위의 점 검증
        st.subheader("검증: 모든 점이 중심으로부터 같은 거리에 있는지 확인")
        distances = []
        for i, point in enumerate(points):
            distance = math.sqrt((point[0] - center_x) ** 2 + (point[1] - center_y) ** 2)
            distances.append(distance)
        
        # 검증 요약 표시
        if distances:
            st.write(f"평균 거리: {sum(distances) / len(distances):.5f} (예상: {radius})")
            st.write(f"최소/최대 거리: {min(distances):.5f} / {max(distances):.5f}")
            st.write(f"거리 표준 편차: {np.std(distances):.8f}")

with tab2:
    # 사이드바에 랜덤 반지름 도형 매개변수 입력 위젯 생성
    st.sidebar.header("랜덤 반지름 도형 설정")
    center_x_rand = st.sidebar.number_input("중심 X좌표", value=0.0, step=0.5, key="cx2")
    center_y_rand = st.sidebar.number_input("중심 Y좌표", value=0.0, step=0.5, key="cy2")
    
    # 반지름 범위 설정
    radius_min = st.sidebar.slider("최소 반지름", min_value=0.1, max_value=8.0, value=3.0, step=0.1, key="rmin")
    radius_max = st.sidebar.slider("최대 반지름", min_value=0.5, max_value=10.0, value=7.0, step=0.1, key="rmax")
    
    num_points_rand = st.sidebar.slider("점의 갯수", min_value=3, max_value=100, value=20, step=1, key="np2")
    
    # 표시 옵션
    st.sidebar.header("표시 옵션")
    show_center_rand = st.sidebar.checkbox("중심점 표시", value=True, key="sc2")
    show_radii_rand = st.sidebar.checkbox("반지름 표시", value=True, key="sr2")
    show_point_labels_rand = st.sidebar.checkbox("점 라벨 표시", value=False, key="spl2")
    show_triangles_rand = st.sidebar.checkbox("삼각형 표시", value=True, key="st2")
    show_polygon_rand = st.sidebar.checkbox("다각형 표시", value=True, key="sp2")
    
    # 시드값 설정 (같은 시드값을 가지면 같은 랜덤 결과가 나옴)
    random_seed = st.sidebar.number_input("랜덤 시드", value=42, min_value=0, max_value=1000, step=1, key="seed")
    random.seed(random_seed)
    
    # 랜덤 반지름으로 점들 생성 함수
    def generate_random_radius_points(center_x, center_y, min_radius, max_radius, num_points):
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            radius = random.uniform(min_radius, max_radius)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append((x, y, angle, radius))  # 각도와 반지름 정보도 저장
        return points
    
    # 점 생성
    rand_points = generate_random_radius_points(center_x_rand, center_y_rand, radius_min, radius_max, num_points_rand)
    rand_points_x = [p[0] for p in rand_points]
    rand_points_y = [p[1] for p in rand_points]
    rand_points_angle = [p[2] for p in rand_points]
    rand_points_radius = [p[3] for p in rand_points]
    
    # 평균 반지름 계산
    avg_radius = sum(rand_points_radius) / len(rand_points_radius)
    
    # 삼각형 넓이 계산
    rand_triangle_areas = []
    for i in range(num_points_rand):
        next_i = (i + 1) % num_points_rand
        area = calculate_triangle_area(
            center_x_rand, center_y_rand, 
            rand_points_x[i], rand_points_y[i], 
            rand_points_x[next_i], rand_points_y[next_i]
        )
        rand_triangle_areas.append(area)
    
    rand_total_triangle_area = sum(rand_triangle_areas)
    
    # 다각형 둘레 계산
    rand_polygon_perimeter = 0
    for i in range(num_points_rand):
        next_i = (i + 1) % num_points_rand
        side = math.sqrt(
            (rand_points_x[next_i] - rand_points_x[i])**2 + 
            (rand_points_y[next_i] - rand_points_y[i])**2
        )
        rand_polygon_perimeter += side

    # 그래프 생성
    fig_rand, ax_rand = plt.subplots(figsize=(8, 8))
    
    # 삼각형 그리기 (선택된 경우)
    if show_triangles_rand:
        for i in range(num_points_rand):
            next_i = (i + 1) % num_points_rand
            triangle = plt.Polygon([
                (center_x_rand, center_y_rand), 
                (rand_points_x[i], rand_points_y[i]), 
                (rand_points_x[next_i], rand_points_y[next_i])
            ], fill=True, alpha=0.1, color='orange')
            ax_rand.add_patch(triangle)
    
    # 다각형 그리기 (선택된 경우)
    if show_polygon_rand:
        polygon_points = [(x, y) for x, y in zip(rand_points_x, rand_points_y)]
        polygon = plt.Polygon(polygon_points, fill=False, edgecolor='purple', linewidth=2)
        ax_rand.add_patch(polygon)
    
    # 점 그리기
    ax_rand.scatter(rand_points_x, rand_points_y, color='red', s=50, zorder=2, label='도형 위의 점들')
    
    # 중심 그리기 (선택된 경우)
    if show_center_rand:
        ax_rand.scatter([center_x_rand], [center_y_rand], color='green', s=100, zorder=3, label='중심')
    
    # 반지름 선 그리기 (선택된 경우)
    if show_radii_rand:
        for i in range(num_points_rand):
            ax_rand.plot([center_x_rand, rand_points_x[i]], [center_y_rand, rand_points_y[i]], 'g-', alpha=0.5)
    
    # 점 라벨 추가 (선택된 경우)
    if show_point_labels_rand:
        for i in range(num_points_rand):
            angle_deg = math.degrees(rand_points_angle[i])
            rad = rand_points_radius[i]
            ax_rand.annotate(f'P{i+1} (r={rad:.1f})', 
                         (rand_points_x[i], rand_points_y[i]), 
                         xytext=(rand_points_x[i] + 0.2, rand_points_y[i] + 0.2))
    
    # 그래프 설정
    ax_rand.set_aspect('equal')
    margin = 1.2
    max_radius = max(rand_points_radius)
    ax_rand.set_xlim(center_x_rand - max_radius * margin, center_x_rand + max_radius * margin)
    ax_rand.set_ylim(center_y_rand - max_radius * margin, center_y_rand + max_radius * margin)
    ax_rand.grid(True)
    ax_rand.set_xlabel('X-축')
    ax_rand.set_ylabel('Y-축')
    ax_rand.set_title(f'중심 ({center_x_rand}, {center_y_rand}), 반지름 범위 [{radius_min}, {radius_max}]인 도형')
    ax_rand.legend()
    
    # 그래프 표시
    st.pyplot(fig_rand)
    
    # 설명 텍스트 추가
    st.markdown("""
    ## 원이 아닌 도형
    이 탭에서는 중심에서 점까지의 거리(반지름)가 일정하지 않은 도형을 보여줍니다.
    
    ## 원의 정의와 비교
    - 원은 중심에서부터 **모든 점까지의 거리가 동일**해야 합니다.
    - 이 도형은 반지름이 일정하지 않으므로 **원이 아닙니다**.
    - 이런 도형은 타원이나 다른 비정형 도형에 가깝습니다.
    """)
    
    # 반지름 정보 표시
    st.subheader("반지름 정보")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**최소 반지름:** {min(rand_points_radius):.3f}")
        st.write(f"**최대 반지름:** {max(rand_points_radius):.3f}")
        st.write(f"**평균 반지름:** {avg_radius:.3f}")
    
    with col2:
        radius_stddev = np.std(rand_points_radius)
        st.write(f"**반지름 표준편차:** {radius_stddev:.3f}")
        st.write(f"**반지름 변동계수:** {(radius_stddev/avg_radius*100):.2f}%")
    
    # 반지름 히스토그램
    st.subheader("반지름 분포")
    hist_fig, hist_ax = plt.subplots(figsize=(8, 4))
    hist_ax.hist(rand_points_radius, bins=10, color='skyblue', edgecolor='black')
    hist_ax.set_xlabel('반지름 길이')
    hist_ax.set_ylabel('빈도')
    hist_ax.set_title('반지름 분포 히스토그램')
    hist_ax.axvline(avg_radius, color='red', linestyle='dashed', linewidth=2, label=f'평균: {avg_radius:.2f}')
    hist_ax.legend()
    st.pyplot(hist_fig)
    
    # 원과의 차이점 설명
    st.subheader("원과의 차이점")
    st.write("아래는 이 도형이 원과 어떻게 다른지를 보여줍니다:")
    
    # 원의 예상 넓이 (평균 반지름 기준)
    expected_circle_area = math.pi * avg_radius ** 2
    
    # 원의 예상 둘레 (평균 반지름 기준)
    expected_circle_perimeter = 2 * math.pi * avg_radius
    
    # 넓이 비교
    st.write("### 넓이 비교")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**삼각형들의 총 넓이:** {rand_total_triangle_area:.3f}")
        st.write(f"**평균 반지름으로 계산한 원의 넓이:** {expected_circle_area:.3f}")
    
    with col2:
        area_error = abs(rand_total_triangle_area - expected_circle_area)
        area_error_percent = (area_error / expected_circle_area) * 100
        st.write(f"**오차:** {area_error:.3f}")
        st.write(f"**오차율:** {area_error_percent:.3f}%")
    
    # 둘레 비교
    st.write("### 둘레 비교")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**다각형의 둘레:** {rand_polygon_perimeter:.3f}")
        st.write(f"**평균 반지름으로 계산한 원의 둘레:** {expected_circle_perimeter:.3f}")
    
    with col2:
        perimeter_error = abs(rand_polygon_perimeter - expected_circle_perimeter)
        perimeter_error_percent = (perimeter_error / expected_circle_perimeter) * 100
        st.write(f"**오차:** {perimeter_error:.3f}")
        st.write(f"**오차율:** {perimeter_error_percent:.3f}%")
    
    st.write("""
    ### 결론
    이 도형은 중심으로부터의 거리가 일정하지 않기 때문에 원의 정의를 만족하지 않습니다.
    위의 분석에서 볼 수 있듯이, 이 도형의 면적과 둘레는 같은 평균 반지름을 가진 원과 상당한 차이가 있습니다.
    원의 정의에서 '모든 점들이 한 점(중심)으로부터 같은 거리에 있다'는 조건이 얼마나 중요한지 보여줍니다.
    """)
    
    # 원 위의 점 정보 간략화 (토글 가능)
    with st.expander("도형 위의 점 정보 (클릭하여 펼치기)"):
        # 점들의 좌표 표시
        st.subheader("도형 위의 점들의 좌표")
        point_data = []
        for i, point in enumerate(rand_points):
            angle_deg = math.degrees(point[2])
            point_data.append({
                "점": f"P{i+1}",
                "X": round(point[0], 3),
                "Y": round(point[1], 3),
                "반지름": round(point[3], 3),
                "각도 (도)": f"{angle_deg:.2f}°"
            })
        st.table(point_data)
