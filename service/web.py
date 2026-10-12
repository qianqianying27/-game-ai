"""内置的极简对局前端(单文件 HTML,零构建)。

人玩家执 X(玩家 1),AI 执 O(玩家 2)。点击列落子,前端调用
``POST /v1/move`` 获取 AI 应手。随页面内联了胜负判定,便于离线演示。
"""
from __future__ import annotations

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Game AI · 四子棋对战</title>
  <style>
    :root { --cell: 60px; }
    * { box-sizing: border-box; }
    body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
           background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px;
           display: flex; flex-direction: column; align-items: center; gap: 16px; }
    h1 { font-size: 20px; font-weight: 600; margin: 0; }
    .status { min-height: 24px; font-size: 15px; color: #94a3b8; }
    .board { display: grid; grid-template-columns: repeat(7, var(--cell));
             grid-auto-rows: var(--cell); gap: 8px; background: #1e293b;
             padding: 12px; border-radius: 12px; }
    .cell { border-radius: 50%; background: #334155; cursor: pointer;
            transition: transform .08s; }
    .cell:hover { transform: scale(1.05); }
    .cell.p1 { background: #38bdf8; }
    .cell.p2 { background: #fb7185; }
    .btn { background: #0ea5e9; color: #fff; border: 0; padding: 10px 18px;
           border-radius: 8px; cursor: pointer; font-size: 14px; }
    .btn:hover { background: #0284c7; }
  </style>
</head>
<body>
  <h1>🎮 四子棋对战 <span style="color:#38bdf8">你(X)</span> vs <span style="color:#fb7185">AI(O)</span></h1>
  <div class="status" id="status">轮到你落子,点击任意列。</div>
  <div class="board" id="board"></div>
  <button class="btn" onclick="reset()">重新开始</button>

  <script>
    const ROWS = 6, COLS = 7;
    let board, over;

    function reset() {
      board = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
      over = false;
      document.getElementById("status").textContent = "轮到你落子,点击任意列。";
      draw();
    }

    function drop(col, player) {
      for (let r = ROWS - 1; r >= 0; r--) {
        if (board[r][col] === 0) { board[r][col] = player; return r; }
      }
      return -1;
    }

    function checkWin(player) {
      const dirs = [[0,1],[1,0],[1,1],[1,-1]];
      for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
        if (board[r][c] !== player) continue;
        for (const [dr, dc] of dirs) {
          let n = 1, rr = r + dr, cc = c + dc;
          while (rr >= 0 && rr < ROWS && cc >= 0 && cc < COLS && board[rr][cc] === player) {
            n++; rr += dr; cc += dc;
          }
          if (n >= 4) return true;
        }
      }
      return false;
    }

    function isFull() { return board[0].every(v => v !== 0); }

    async function aiMove() {
      document.getElementById("status").textContent = "AI 思考中…";
      const resp = await fetch("/v1/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ board, player: 2 })
      });
      if (!resp.ok) { document.getElementById("status").textContent = "AI 请求失败"; return; }
      const { action } = await resp.json();
      drop(action, 2);
      if (checkWin(2)) { setStatus("AI 获胜 😵"); over = true; }
      else if (isFull()) { setStatus("平局 🤝"); over = true; }
      else setStatus("轮到你落子。");
      draw();
    }

    function setStatus(t) { document.getElementById("status").textContent = t; }

    async function onClick(col) {
      if (over) return;
      if (drop(col, 1) < 0) return;
      draw();
      if (checkWin(1)) { setStatus("你获胜 🎉"); over = true; return; }
      if (isFull()) { setStatus("平局 🤝"); over = true; return; }
      await aiMove();
    }

    function draw() {
      const el = document.getElementById("board");
      el.innerHTML = "";
      for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
        const cell = document.createElement("div");
        cell.className = "cell" + (board[r][c] === 1 ? " p1" : board[r][c] === 2 ? " p2" : "");
        cell.onclick = () => onClick(c);
        el.appendChild(cell);
      }
    }

    reset();
  </script>
</body>
</html>
"""
