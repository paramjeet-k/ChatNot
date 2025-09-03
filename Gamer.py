# app.py
import streamlit as st
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import graphviz
import pandas as pd

# =========================
# Core Tic-Tac-Toe logic
# =========================
WIN_LINES = [
    (0,1,2),(3,4,5),(6,7,8),   # rows
    (0,3,6),(1,4,7),(2,5,8),   # cols
    (0,4,8),(2,4,6)            # diagonals
]

def check_winner(board: List[str]) -> Optional[str]:
    """Return 'X', 'O', 'Draw', or None."""
    for a,b,c in WIN_LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    if all(board):
        return "Draw"
    return None

def available_moves(board: List[str]) -> List[int]:
    return [i for i, v in enumerate(board) if v == ""]

def make_move(board: List[str], idx: int, player: str) -> List[str]:
    nb = board[:]
    nb[idx] = player
    return nb

def next_player(p: str) -> str:
    return "O" if p == "X" else "X"

def board_str(board: List[str]) -> str:
    return "".join(v if v else "_" for v in board)

# =========================
# Search tree structures
# =========================
@dataclass
class Node:
    id: int
    board: List[str]
    turn: str            # player to move at this node
    parent: Optional[int]
    move: Optional[int]  # square index (0..8) taken from parent to here
    depth: int
    winner: Optional[str]  # terminal outcome at this node (X/O/Draw/None)
    order: int           # expansion order (1..N)

def expand_tree(
    root_board: List[str],
    root_turn: str,
    method: str = "BFS",
    depth_limit: int = 6,
    max_nodes: int = 2000
) -> Tuple[Dict[int, Node], List[Tuple[int, int]], Dict[str, int]]:
    """
    Build a partial game tree (from current position).
    Returns nodes dict, edges list, and stats.
    """
    nodes: Dict[int, Node] = {}
    edges: List[Tuple[int, int]] = []
    nid = 0
    order = 0

    # create root
    root = Node(
        id=nid, board=root_board[:], turn=root_turn,
        parent=None, move=None, depth=0,
        winner=check_winner(root_board), order=0
    )
    nodes[nid] = root

    # frontier
    if method.upper() == "BFS":
        frontier = deque([nid])
        pop = frontier.popleft
        push = frontier.append
    else:  # DFS (LIFO)
        frontier = [nid]
        pop = frontier.pop
        push = frontier.append

    expanded = 0
    term_x = term_o = term_d = 0

    while frontier and len(nodes) < max_nodes:
        cur_id = pop()
        cur = nodes[cur_id]

        # mark expansion order
        order += 1
        cur.order = order
        expanded += 1

        # tally terminals
        if cur.winner == "X": term_x += 1
        if cur.winner == "O": term_o += 1
        if cur.winner == "Draw": term_d += 1

        # stop at terminal or depth limit
        if cur.winner is not None or cur.depth >= depth_limit:
            continue

        # expand children
        for m in available_moves(cur.board):
            nid += 1
            child_board = make_move(cur.board, m, cur.turn)
            child = Node(
                id=nid, board=child_board, turn=next_player(cur.turn),
                parent=cur.id, move=m, depth=cur.depth + 1,
                winner=check_winner(child_board), order=0
            )
            nodes[nid] = child
            edges.append((cur.id, nid))
            push(nid)

    stats = {
        "expanded": expanded,
        "total_nodes": len(nodes),
        "X_wins": term_x,
        "O_wins": term_o,
        "draws": term_d
    }
    return nodes, edges, stats

# =========================
# Graphviz rendering
# =========================
def html_board_label(n: Node) -> str:
    """
    HTML-like label for Graphviz with a compact 3x3 grid and a header row.
    """
    def cell(v):  # use nbsp for empty
        return v if v else "&nbsp;"
    status = (f"{n.winner} wins" if n.winner in ("X","O")
              else ("Draw" if n.winner=="Draw" else f"{n.turn} to move"))
    header = f"#{n.id} • d{n.depth} • {status} • @{n.order}"
    b = n.board
    label = f"""
    <
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
      <TR><TD COLSPAN="3"><B>{header}</B></TD></TR>
      <TR><TD>{cell(b[0])}</TD><TD>{cell(b[1])}</TD><TD>{cell(b[2])}</TD></TR>
      <TR><TD>{cell(b[3])}</TD><TD>{cell(b[4])}</TD><TD>{cell(b[5])}</TD></TR>
      <TR><TD>{cell(b[6])}</TD><TD>{cell(b[7])}</TD><TD>{cell(b[8])}</TD></TR>
    </TABLE>
    >
    """
    return label

def color_for(n: Node) -> str:
    if n.parent is None:      # root
        return "#DBEAFE"      # light blue
    if n.winner == "X":
        return "#C8FACC"      # pale green
    if n.winner == "O":
        return "#FFD1D1"      # pale red
    if n.winner == "Draw":
        return "#EEEEEE"      # grey
    return "#FFFFFF"          # white

def path_to_root(nodes: Dict[int, Node], target_id: int) -> List[int]:
    path = []
    cur = nodes.get(target_id)
    while cur is not None:
        path.append(cur.id)
        cur = nodes.get(cur.parent) if cur.parent is not None else None
    return list(reversed(path))

def render_graph(
    nodes: Dict[int, Node],
    edges: List[Tuple[int, int]],
    highlight_path: Optional[List[int]] = None
) -> graphviz.Digraph:
    if highlight_path is None:
        highlight_path = []

    path_edges = set()
    for i in range(len(highlight_path)-1):
        path_edges.add((highlight_path[i], highlight_path[i+1]))

    dot = graphviz.Digraph(
        "ttt",
        node_attr={"shape": "box", "fontname": "Helvetica"},
        graph_attr={"rankdir": "TB", "splines": "true"}
    )
    # nodes
    for n in nodes.values():
        dot.node(
            str(n.id),
            label=html_board_label(n),
            style="filled",
            fillcolor=color_for(n)
        )
    # edges
    for u, v in edges:
        # label with human-friendly 1..9 cell index
        mv = nodes[v].move
        attrs = {}
        if (u, v) in path_edges:
            attrs = {"color": "#2563EB", "penwidth": "3"}  # blue highlight
        dot.edge(str(u), str(v), label=str(mv+1), **attrs)
    return dot

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Tic Tac Toe — BFS/DFS Game Tree", page_icon="🎮", layout="wide")
st.title("🎮 Tic Tac Toe — Play + BFS/DFS Game-Tree Visualizer")
st.caption("Play the game and explore the game tree. Change the depth/algorithm and the tree redraws instantly. Pick any node to load that position onto the board!")

# --- Session state ---
if "board" not in st.session_state:
    st.session_state.board = ["" for _ in range(9)]
    st.session_state.turn = "X"
    st.session_state.winner = None

# --- Sidebar: controls ---
with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Reset Game"):
        st.session_state.board = ["" for _ in range(9)]
        st.session_state.turn = "X"
        st.session_state.winner = None
        st.rerun()

    st.subheader("Tree Builder")
    method = st.radio("Search Method", ["BFS", "DFS"], horizontal=True)
    depth = st.slider("Depth limit (plies)", 1, 9, 6, 1, help="Number of half-moves from the current position.")
    max_nodes = st.slider("Max nodes", 50, 5000, 1500, 50)

# --- Board UI (immersive) ---
st.markdown("""
<style>
div.stButton > button {
  height: 100px;
  font-size: 36px;
}
</style>
""", unsafe_allow_html=True)

left, right = st.columns([1,1])
with left:
    st.subheader("Game Board")
    grid_cols = st.columns(3, gap="small")

    for i in range(9):
        c = grid_cols[i % 3]
        with c:
            label = st.session_state.board[i] if st.session_state.board[i] else " "
            disabled = st.session_state.board[i] != "" or st.session_state.winner is not None
            if st.button(label, key=f"sq_{i}", use_container_width=True, disabled=disabled):
                st.session_state.board[i] = st.session_state.turn
                st.session_state.winner = check_winner(st.session_state.board)
                if not st.session_state.winner:
                    st.session_state.turn = next_player(st.session_state.turn)
                st.rerun()

    # status line
    if st.session_state.winner:
        if st.session_state.winner == "Draw":
            st.success("🤝 It's a draw!")
        else:
            st.success(f"🎉 Player **{st.session_state.winner}** wins!")
    else:
        st.info(f"Turn: **{st.session_state.turn}**")

# --- Build tree from current position ---
nodes, edges, stats = expand_tree(
    root_board=st.session_state.board,
    root_turn=st.session_state.turn,
    method=method,
    depth_limit=depth,
    max_nodes=max_nodes
)

# --- Right panel: visualizer + explorer ---
with right:
    st.subheader("Game-Tree Visualizer")
    # selection to highlight a node path, and optional load
    selectable_ids = sorted(nodes.keys())
    select_id = st.selectbox(
        "Highlight a node (path from root will be emphasized):",
        selectable_ids,
        index=0,
        format_func=lambda i: f"#{i} — d{nodes[i].depth} — {board_str(nodes[i].board)}"
    )
    path = path_to_root(nodes, select_id)
    dot = render_graph(nodes, edges, highlight_path=path)
    st.graphviz_chart(dot, use_container_width=True)

    # load button to replace board with chosen node
    st.markdown("**Load the selected node onto the game board**")
    if st.button("⬇️ Load position"):
        nb = nodes[select_id].board[:]
        st.session_state.board = nb
        st.session_state.turn = nodes[select_id].turn
        st.session_state.winner = check_winner(nb)
        st.rerun()

# --- Explorer table and stats ---
st.markdown("---")
colA, colB = st.columns([2,1], vertical_alignment="center")

with colA:
    st.subheader("Node Explorer")
    df = pd.DataFrame({
        "id": [n.id for n in nodes.values()],
        "depth": [n.depth for n in nodes.values()],
        "to_move": [n.turn for n in nodes.values()],
        "winner": [n.winner if n.winner else "" for n in nodes.values()],
        "parent": [n.parent for n in nodes.values()],
        "move(1..9)": [(n.move + 1) if n.move is not None else None for n in nodes.values()],
        "expand_order": [n.order for n in nodes.values()],
        "board": [board_str(n.board) for n in nodes.values()],
    }).sort_values(["depth","expand_order","id"], kind="stable")
    st.dataframe(df, use_container_width=True, height=320)

with colB:
    st.subheader("Stats")
    st.metric("Nodes expanded", stats["expanded"])
    st.metric("Total nodes", stats["total_nodes"])
    c1, c2, c3 = st.columns(3)
    c1.metric("X wins", stats["X_wins"])
    c2.metric("O wins", stats["O_wins"])
    c3.metric("Draws", stats["draws"])

st.caption("Tip: Change method/depth/max nodes, pick a node to highlight its path, or load it to the board and keep playing.")
