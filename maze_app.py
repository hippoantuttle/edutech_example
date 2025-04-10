import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# 페이지 기본 설정
st.set_page_config(
    page_title="미로 제작 툴",
    layout="wide"
)

# 앱 제목
st.title("학생을 위한 미로 제작 툴")

# 사이드바에 미로 크기 입력 받기
st.sidebar.header("미로 설정")
rows = st.sidebar.number_input("행 수", min_value=3, max_value=30, value=10)
cols = st.sidebar.number_input("열 수", min_value=3, max_value=30, value=10)

# 미로 초기화 (0은 통로, 1은 벽)
if 'maze' not in st.session_state:
    st.session_state.maze = np.zeros((rows, cols), dtype=int)

if 'start' not in st.session_state:
    st.session_state.start = None

if 'end' not in st.session_state:
    st.session_state.end = None

# 행동 선택
action = st.sidebar.radio(
    "동작 선택",
    ["시작점 설정", "도착점 설정", "벽 그리기", "통로 그리기"]
)

# 미로 크기가 변경되면 미로 재설정
if st.session_state.maze.shape != (rows, cols):
    st.session_state.maze = np.zeros((rows, cols), dtype=int)
    st.session_state.start = None
    st.session_state.end = None

# 미로 시각화 함수
def visualize_maze():
    fig, ax = plt.subplots(figsize=(10, 10 * rows/cols))
    
    # 미로 표시
    cmap = plt.cm.binary
    ax.imshow(st.session_state.maze, cmap=cmap)
    
    # 시작점 표시
    if st.session_state.start:
        y, x = st.session_state.start
        ax.plot(x, y, 'go', markersize=15, label='Start Point')
    
    # 도착점 표시
    if st.session_state.end:
        y, x = st.session_state.end
        ax.plot(x, y, 'ro', markersize=15, label='End Point')
    
    # 그리드 표시
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks(np.arange(0, cols, 1))
    ax.set_yticks(np.arange(0, rows, 1))
    
    # 추가 설정
    ax.set_xlim(-0.5, cols-0.5)
    ax.set_ylim(rows-0.5, -0.5)
    if st.session_state.start or st.session_state.end:
        ax.legend()
    
    return fig

# 미로 좌표 입력
st.sidebar.header("좌표 입력")
row_input = st.sidebar.number_input("행 (y):", min_value=0, max_value=rows-1, value=0)
col_input = st.sidebar.number_input("열 (x):", min_value=0, max_value=cols-1, value=0)

# 설정 버튼
if st.sidebar.button("설정"):
    pos = (row_input, col_input)
    
    if action == "시작점 설정":
        if st.session_state.maze[pos] == 0:  # 벽이 아닌 경우만 시작점 설정
            st.session_state.start = pos
            st.success(f"시작점이 ({pos[0]}, {pos[1]})에 설정되었습니다.")
        else:
            st.error("벽에는 시작점을 설정할 수 없습니다.")
    
    elif action == "도착점 설정":
        if st.session_state.maze[pos] == 0:  # 벽이 아닌 경우만 도착점 설정
            st.session_state.end = pos
            st.success(f"도착점이 ({pos[0]}, {pos[1]})에 설정되었습니다.")
        else:
            st.error("벽에는 도착점을 설정할 수 없습니다.")
    
    elif action == "벽 그리기":
        st.session_state.maze[pos] = 1
        # 시작점이나 도착점이 벽과 겹치면 제거
        if st.session_state.start == pos:
            st.session_state.start = None
        if st.session_state.end == pos:
            st.session_state.end = None
    
    elif action == "통로 그리기":
        st.session_state.maze[pos] = 0

# 미로 시각화
st.subheader("미로")
fig = visualize_maze()
st.pyplot(fig)

# 미로 검증 알고리즘
def is_solvable(maze, start, end):
    if start is None or end is None:
        return False
    
    # BFS 알고리즘
    rows, cols = maze.shape
    visited = np.zeros_like(maze, dtype=bool)
    queue = deque([start])
    visited[start] = True
    
    # 상하좌우 이동 방향
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        r, c = queue.popleft()
        
        # 도착점에 도달했는지 확인
        if (r, c) == end:
            return True
        
        # 네 방향 확인
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            # 범위 내에 있고, 벽이 아니며, 방문하지 않은 위치인지 확인
            if (0 <= nr < rows and 0 <= nc < cols and 
                maze[nr, nc] == 0 and not visited[nr, nc]):
                queue.append((nr, nc))
                visited[nr, nc] = True
    
    return False

# 경로 찾기 알고리즘
def find_path(maze, start, end):
    if start is None or end is None:
        return []
    
    rows, cols = maze.shape
    visited = np.zeros_like(maze, dtype=bool)
    parent = {}
    queue = deque([start])
    visited[start] = True
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        r, c = queue.popleft()
        
        if (r, c) == end:
            # 경로 역추적
            path = []
            current = end
            while current != start:
                path.append(current)
                current = parent[current]
            path.append(start)
            return path[::-1]  # 경로 뒤집기
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if (0 <= nr < rows and 0 <= nc < cols and 
                maze[nr, nc] == 0 and not visited[nr, nc]):
                queue.append((nr, nc))
                visited[nr, nc] = True
                parent[(nr, nc)] = (r, c)
    
    return []

# 검증 버튼
if st.button("Go - 미로 검증"):
    if st.session_state.start is None or st.session_state.end is None:
        st.error("시작점과 도착점을 모두 설정해야 합니다.")
    else:
        solvable = is_solvable(st.session_state.maze, st.session_state.start, st.session_state.end)
        
        if solvable:
            st.success("미로가 해결 가능합니다!")
            
            # 경로 찾기
            path = find_path(st.session_state.maze, st.session_state.start, st.session_state.end)
            
            # 경로를 시각화
            fig, ax = plt.subplots(figsize=(10, 10 * rows/cols))
            ax.imshow(st.session_state.maze, cmap='binary')
            
            # 경로 그리기
            if path:
                path_y, path_x = zip(*path)
                ax.plot(path_x, path_y, 'b-', linewidth=2)
                ax.plot(path_x, path_y, 'bo', markersize=8)
            
            # 시작점 및 도착점 표시
            y, x = st.session_state.start
            ax.plot(x, y, 'go', markersize=15, label='시작점')
            
            y, x = st.session_state.end
            ax.plot(x, y, 'ro', markersize=15, label='도착점')
            
            # 그리드 및 기타 설정
            ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
            ax.grid(which="minor", color="gray", linestyle='-', linewidth=1)
            ax.tick_params(which="minor", size=0)
            ax.set_xticks(np.arange(0, cols, 1))
            ax.set_yticks(np.arange(0, rows, 1))
            ax.set_xlim(-0.5, cols-0.5)
            ax.set_ylim(rows-0.5, -0.5)
            ax.legend()
            
            st.subheader("미로 해결 경로")
            st.pyplot(fig)
        else:
            st.error("미로가 해결 불가능합니다. 경로를 확인해주세요.")

# 미로 초기화 버튼
if st.sidebar.button("미로 초기화"):
    st.session_state.maze = np.zeros((rows, cols), dtype=int)
    st.session_state.start = None
    st.session_state.end = None
    st.experimental_rerun()
