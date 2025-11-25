
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, Type } from "@google/genai";

// --- Constants ---
const BOARD_SIZE = 15;
const PLAYER_BLACK = 1; // Stone Color Black (VOID)
const PLAYER_WHITE = 2; // Stone Color White (LIGHT)
const EMPTY = 0;

type CellValue = 0 | 1 | 2;
type BoardState = CellValue[][];
type Difficulty = 'EASY' | 'MEDIUM' | 'SUPER_STRONG' | 'HARD';
type GameMode = 'PVP' | 'PVE';

// --- Helper Logic ---

// Check for ANY line length >= 5. Returns ALL stones in that line.
const checkLongLines = (board: BoardState, lastRow: number, lastCol: number, player: CellValue) => {
  const directions = [
    [0, 1],   // Horizontal
    [1, 0],   // Vertical
    [1, 1],   // Diagonal \
    [1, -1]   // Diagonal /
  ];

  let allStonesInLine: { r: number, c: number }[] = [];

  for (const [dx, dy] of directions) {
    let count = 1;
    let lineCells = [{ r: lastRow, c: lastCol }];

    // Check forward (positive direction)
    let i = 1;
    while (true) {
      const r = lastRow + dx * i;
      const c = lastCol + dy * i;
      if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE && board[r][c] === player) {
        count++;
        lineCells.push({ r, c });
        i++;
      } else {
        break;
      }
    }

    // Check backward (negative direction)
    let j = 1;
    while (true) {
      const r = lastRow - dx * j;
      const c = lastCol - dy * j;
      if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE && board[r][c] === player) {
        count++;
        lineCells.push({ r, c });
        j++;
      } else {
        break;
      }
    }

    // If line length is 5 OR MORE, add to removal list
    if (count >= 5) {
      allStonesInLine = [...allStonesInLine, ...lineCells];
    }
  }

  // Remove duplicates
  const uniqueStones = Array.from(new Set(allStonesInLine.map(s => `${s.r},${s.c}`)))
                            .map(str => {
                                const [r, c] = str.split(',').map(Number);
                                return { r, c };
                            });

  return uniqueStones.length > 0 ? uniqueStones : null;
};

const isBoardFull = (board: BoardState): boolean => {
    for(let r=0; r<BOARD_SIZE; r++){
        for(let c=0; c<BOARD_SIZE; c++){
            if(board[r][c] === EMPTY) return false;
        }
    }
    return true;
};

const countStones = (board: BoardState) => {
    let black = 0;
    let white = 0;
    board.forEach(row => row.forEach(cell => {
        if (cell === PLAYER_BLACK) black++;
        if (cell === PLAYER_WHITE) white++;
    }));
    return { black, white };
};

const getEmptyNeighbors = (board: BoardState, distance = 2): { r: number, c: number }[] => {
  const neighbors = new Set<string>();
  const moves: { r: number, c: number }[] = [];

  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      if (board[r][c] !== EMPTY) {
        for (let dr = -distance; dr <= distance; dr++) {
          for (let dc = -distance; dc <= distance; dc++) {
            const nr = r + dr;
            const nc = c + dc;
            if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE && board[nr][nc] === EMPTY) {
              const key = `${nr},${nc}`;
              if (!neighbors.has(key)) {
                neighbors.add(key);
                moves.push({ r: nr, c: nc });
              }
            }
          }
        }
      }
    }
  }
  if (moves.length === 0) return [{ r: Math.floor(BOARD_SIZE / 2), c: Math.floor(BOARD_SIZE / 2) }];
  return moves;
};

// --- AI Strategies ---

// Level 1: Easy
const getEasyMove = (board: BoardState, aiPlayer: CellValue) => {
  const moves = getEmptyNeighbors(board, 1);
  if (moves.length === 0) return { r: 7, c: 7 };
  const randomIdx = Math.floor(Math.random() * moves.length);
  return moves[randomIdx];
};

// Level 2: Medium (Heuristic)
const getMediumMove = (board: BoardState, aiPlayer: CellValue) => {
    const moves = getEmptyNeighbors(board, 1);
    let bestScore = -Infinity;
    let bestMove = moves[0];

    const evaluate = (r: number, c: number, player: CellValue, currentBoard: BoardState) => {
        const stonesExploded = checkLongLines(currentBoard, r, c, player);
        if (stonesExploded) return -1000 * stonesExploded.length; 

        let score = 0;
        const directions = [[0, 1], [1, 0], [1, 1], [1, -1]];
        
        for (const [dx, dy] of directions) {
            let countAI = 0;
            let openEndsAI = 0;
            for(let k=1; k<5; k++) {
                const nr = r + dx*k, nc = c + dy*k;
                if(nr>=0 && nr<15 && nc>=0 && nc<15) {
                    if(currentBoard[nr][nc] === player) countAI++;
                    else if(currentBoard[nr][nc] === EMPTY) { openEndsAI++; break; }
                    else break;
                }
            }
            for(let k=1; k<5; k++) {
                const nr = r - dx*k, nc = c - dy*k;
                if(nr>=0 && nr<15 && nc>=0 && nc<15) {
                    if(currentBoard[nr][nc] === player) countAI++;
                    else if(currentBoard[nr][nc] === EMPTY) { openEndsAI++; break; }
                    else break;
                }
            }
            if (countAI === 3 && openEndsAI > 0) score += 50; 
            if (countAI === 2 && openEndsAI > 0) score += 10;
        }
        score += Math.random() * 5; 
        return score;
    }

    for(const move of moves) {
        const tempBoard = board.map(row => [...row]);
        tempBoard[move.r][move.c] = aiPlayer;
        const score = evaluate(move.r, move.c, aiPlayer, tempBoard);
        if(score > bestScore) {
            bestScore = score;
            bestMove = move;
        }
    }
    return bestMove;
};

// Level 3: Super Strong (Local - Minimax Shallow)
const getSuperStrongMove = (board: BoardState, aiPlayer: CellValue) => {
    const moves = getEmptyNeighbors(board, 1);
    const candidates = [];
    
    for (const move of moves) {
        const tempBoard = board.map(row => [...row]);
        tempBoard[move.r][move.c] = aiPlayer;
        const exploded = checkLongLines(tempBoard, move.r, move.c, aiPlayer);
        if (exploded) {
            candidates.push({ move, score: -10000 * exploded.length });
        } else {
            candidates.push({ move, score: 0 });
        }
    }

    const opponent = aiPlayer === PLAYER_BLACK ? PLAYER_WHITE : PLAYER_BLACK;
    
    for (let i = 0; i < candidates.length; i++) {
        if (candidates[i].score < -5000) continue; 

        const { r, c } = candidates[i].move;
        const tempBoard = board.map(row => [...row]);
        tempBoard[r][c] = aiPlayer;

        let moveScore = 0;
        
        const directions = [[0, 1], [1, 0], [1, 1], [1, -1]];
        for (const [dx, dy] of directions) {
            let lineCount = 1;
            if (r+dx < 15 && r+dx >= 0 && tempBoard[r+dx][c+dy] === aiPlayer) lineCount++;
            if (r-dx < 15 && r-dx >= 0 && tempBoard[r-dx][c-dy] === aiPlayer) lineCount++;
            
            if (lineCount === 3) moveScore -= 50; 
            if (lineCount === 4) moveScore -= 200; 
            if (lineCount === 2) moveScore += 10; 
        }

        let liberties = 0;
        for(let dr=-1; dr<=1; dr++){
            for(let dc=-1; dc<=1; dc++){
                if(dr===0 && dc===0) continue;
                if (r+dr>=0 && r+dr<15 && c+dc>=0 && c+dc<15 && tempBoard[r+dr][c+dc] === EMPTY) {
                    liberties++;
                }
            }
        }
        moveScore += liberties * 5; 
        candidates[i].score += moveScore + Math.random() * 2;
    }
    candidates.sort((a, b) => b.score - a.score);
    return candidates[0].move;
};

// Level 4: Extreme (Gemini API)
const getGeminiMove = async (board: BoardState, aiPlayer: CellValue, scoreSelf: number, scoreOpponent: number): Promise<{ r: number, c: number } | null> => {
  try {
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    const boardStr = JSON.stringify(board);
    const playerColorName = aiPlayer === PLAYER_BLACK ? "Void (1)" : "Light (2)";
    
    const prompt = `
      You are playing **Caro Nổ (Explosive Gomoku)**.
      Board: 15x15.
      
      **CRITICAL SCORING RULE:**
      If you form a line of **5 OR MORE** stones (5, 6, 7, 8...):
      1. Your OPPONENT gets points equal to the number of stones.
      2. Those stones are REMOVED from the board.
      
      **GOAL:** Maximize YOUR score (${scoreSelf}) relative to the opponent (${scoreOpponent}). 
      
      **STRATEGY:** 
      - **DO NOT** make a line of 5+ stones yourself unless it gives you a massive tactical advantage by clearing space.
      - **FORCE** the opponent to complete a line of 5+ stones.
      - Avoid building long lines (3 or 4) unless you are sure they won't become 5.
      
      Swap Rule: Sides swap every 15 moves. currently playing **${playerColorName}**.
      
      Current Board (JSON): ${boardStr}
      
      Return best move [row, col].
      Response JSON format: { "row": number, "col": number, "reasoning": string }
    `;

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            row: { type: Type.INTEGER },
            col: { type: Type.INTEGER },
            reasoning: { type: Type.STRING }
          },
          required: ["row", "col"]
        }
      }
    });

    const result = JSON.parse(response.text);
    console.log("Gemini Reasoning:", result.reasoning);
    
    if (result.row >= 0 && result.row < 15 && result.col >= 0 && result.col < 15 && board[result.row][result.col] === EMPTY) {
      return { r: result.row, c: result.col };
    }
    throw new Error("Invalid Gemini move");

  } catch (error) {
    console.error("Gemini failed, falling back to Strong AI:", error);
    return getSuperStrongMove(board, aiPlayer);
  }
};

const getGameResult = (
    currentScores: { p1: number, p2: number }, 
    finalBoard: BoardState, 
    currentHumanColor: CellValue, 
    gameMode: GameMode
) => {
    const sP1 = currentScores.p1;
    const sP2 = currentScores.p2;
    const p1Name = gameMode === 'PVE' ? "BẠN" : "NGƯỜI CHƠI 1";
    const p2Name = gameMode === 'PVE' ? "MÁY" : "NGƯỜI CHƠI 2";
    
    if (sP1 > sP2) return { winner: p1Name, reason: "Điểm cao hơn" };
    if (sP2 > sP1) return { winner: p2Name, reason: "Điểm cao hơn" };

    const { black, white } = countStones(finalBoard);
    const p1Stones = currentHumanColor === PLAYER_BLACK ? black : white;
    const p2Stones = currentHumanColor === PLAYER_BLACK ? white : black;

    if (p1Stones > p2Stones) {
        const diff = p1Stones - p2Stones;
        return { winner: p2Name, reason: `Hòa điểm, nhưng ${p1Name} còn nhiều hơn ${diff} quân (bị trừ điểm)` };
    }
    if (p2Stones > p1Stones) {
        const diff = p2Stones - p1Stones;
        return { winner: p1Name, reason: `Hòa điểm, nhưng ${p2Name} còn nhiều hơn ${diff} quân (bị trừ điểm)` };
    }

    return { winner: "HÒA", reason: "Hòa điểm và bằng số quân" };
};

const App = () => {
  const [board, setBoard] = useState<BoardState>(Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(EMPTY)));
  const [scores, setScores] = useState({ p1: 0, p2: 0 });
  const [currentPlayer, setCurrentPlayer] = useState<CellValue>(PLAYER_BLACK);
  const [humanColor, setHumanColor] = useState<CellValue>(PLAYER_BLACK);
  const [turnCount, setTurnCount] = useState(0); 
  const [isGameOver, setIsGameOver] = useState(false);
  const [finalResult, setFinalResult] = useState<{ winner: string, reason: string } | null>(null);
  const [gameMode, setGameMode] = useState<GameMode>('PVE');
  const [difficulty, setDifficulty] = useState<Difficulty>('EASY');
  const [isAiThinking, setIsAiThinking] = useState(false);
  const [lastMove, setLastMove] = useState<{r: number, c: number} | null>(null);
  const [showSwapAlert, setShowSwapAlert] = useState(false);
  const [removedStones, setRemovedStones] = useState<{r:number, c:number}[]>([]);
  const [hoveredCell, setHoveredCell] = useState<{r: number, c: number} | null>(null);

  const movesUntilNextSwap = 30 - (turnCount % 30);

  const playSound = (type: 'click' | 'clear') => {};

  const resetGame = () => {
    setBoard(Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(EMPTY)));
    setCurrentPlayer(PLAYER_BLACK);
    setHumanColor(PLAYER_BLACK);
    setTurnCount(0);
    setScores({ p1: 0, p2: 0 });
    setIsGameOver(false);
    setFinalResult(null);
    setLastMove(null);
    setIsAiThinking(false);
    setShowSwapAlert(false);
    setRemovedStones([]);
    setHoveredCell(null);
  };

  const processTurn = useCallback((
      newBoard: BoardState, 
      r: number, 
      c: number, 
      playerWhoMoved: CellValue, 
      currentHumanColor: CellValue,
      baseScores: { p1: number, p2: number },
      currentTurnBase: number 
    ) => {
     
     let updatedBoard = newBoard;
     let currentScores = { ...baseScores };

     const stonesToExplode = checkLongLines(newBoard, r, c, playerWhoMoved);
    
     if (stonesToExplode && stonesToExplode.length > 0) {
       const points = stonesToExplode.length;
       const isP1Move = playerWhoMoved === currentHumanColor;
       
       if (isP1Move) {
           currentScores.p2 += points;
       } else {
           currentScores.p1 += points;
       }
       
       const tempBoard = newBoard.map(row => [...row]);
       stonesToExplode.forEach(cell => {
           tempBoard[cell.r][cell.c] = EMPTY;
       });
       updatedBoard = tempBoard;
       
       setRemovedStones(stonesToExplode);
       setTimeout(() => setRemovedStones([]), 500);
     }
     
     setScores(currentScores);
     setBoard(updatedBoard);

     const p1Name = gameMode === 'PVE' ? "BẠN" : "NGƯỜI CHƠI 1";
     const p2Name = gameMode === 'PVE' ? "MÁY" : "NGƯỜI CHƠI 2";

     if (currentScores.p1 >= 200 && (currentScores.p1 - currentScores.p2) >= 100) {
         setFinalResult({ winner: p1Name, reason: "Thắng áp đảo (Hơn 100 điểm & đạt mốc 200)" });
         setIsGameOver(true);
         return;
     }
     if (currentScores.p2 >= 200 && (currentScores.p2 - currentScores.p1) >= 100) {
         setFinalResult({ winner: p2Name, reason: "Thắng áp đảo (Hơn 100 điểm & đạt mốc 200)" });
         setIsGameOver(true);
         return;
     }
 
     if (isBoardFull(updatedBoard)) {
        const result = getGameResult(currentScores, updatedBoard, currentHumanColor, gameMode);
        setFinalResult(result);
        setIsGameOver(true);
        return;
     }

     const nextTurnCount = currentTurnBase + 1;
     setTurnCount(nextTurnCount);

     const nextPlayerColor = playerWhoMoved === PLAYER_BLACK ? PLAYER_WHITE : PLAYER_BLACK;
     
     let nextHumanColor = currentHumanColor;
     if (nextTurnCount > 0 && nextTurnCount % 30 === 0) {
         nextHumanColor = currentHumanColor === PLAYER_BLACK ? PLAYER_WHITE : PLAYER_BLACK;
         setHumanColor(nextHumanColor);
         setShowSwapAlert(true);
         setTimeout(() => setShowSwapAlert(false), 3000);
     }
 
     setCurrentPlayer(nextPlayerColor);
 
     if (gameMode === 'PVE' && nextPlayerColor !== nextHumanColor && !isBoardFull(updatedBoard)) {
       setIsAiThinking(true);
       setTimeout(() => makeAiMove(updatedBoard, nextPlayerColor, nextHumanColor, currentScores.p2, currentScores.p1, nextTurnCount), 600); 
     } else {
        setIsAiThinking(false);
     }
  }, [gameMode]);

  const handleCellClick = async (r: number, c: number) => {
    if (isGameOver || isAiThinking) return;
    if (gameMode === 'PVE' && currentPlayer !== humanColor) return;
    if (board[r][c] !== EMPTY) return;

    const newBoard = board.map(row => [...row]);
    newBoard[r][c] = currentPlayer;
    
    setBoard(newBoard);
    setLastMove({ r, c });
    playSound('click');

    processTurn(newBoard, r, c, currentPlayer, humanColor, scores, turnCount);
  };

  const makeAiMove = async (currentBoard: BoardState, aiPlayer: CellValue, currentHumanColor: CellValue, aiScore: number, humanScore: number, currentTurnCount: number) => {
    let move: { r: number, c: number } | null = null;

    if (difficulty === 'EASY') {
      move = getEasyMove(currentBoard, aiPlayer);
    } else if (difficulty === 'MEDIUM') {
      await new Promise(resolve => setTimeout(resolve, 300));
      move = getMediumMove(currentBoard, aiPlayer);
    } else if (difficulty === 'SUPER_STRONG') {
      await new Promise(resolve => setTimeout(resolve, 500));
      move = getSuperStrongMove(currentBoard, aiPlayer);
    } else {
      move = await getGeminiMove(currentBoard, aiPlayer, aiScore, humanScore);
    }

    if (move) {
      const nextBoard = currentBoard.map(row => [...row]);
      nextBoard[move.r][move.c] = aiPlayer;
      
      setBoard(nextBoard);
      setLastMove({ r: move.r, c: move.c });
      playSound('click');
      
      const currentScores = { p1: humanScore, p2: aiScore };
      processTurn(nextBoard, move.r, move.c, aiPlayer, currentHumanColor, currentScores, currentTurnCount);
    } else {
        setIsGameOver(true); 
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4">
      <div className="max-w-6xl w-full bg-white rounded-xl shadow-2xl overflow-hidden flex flex-col md:flex-row">
        
        {/* Sidebar */}
        <div className="w-full md:w-1/3 bg-gray-900 text-white p-6 flex flex-col gap-5">
          <div>
            <h1 className="text-3xl font-bold text-yellow-500">Caro Nổ</h1>
            <p className="text-gray-400 text-xs mt-1">Đổi màu quân mỗi 30 lượt</p>
          </div>

          {/* Scoreboard */}
          <div className="grid grid-cols-2 gap-4">
              <div className={`p-3 rounded-lg border relative overflow-hidden ${humanColor === currentPlayer ? 'border-yellow-500 bg-yellow-900/20' : 'border-gray-700 bg-gray-800'}`}>
                  <div className="flex justify-between items-start mb-2">
                      <div className="text-xs text-gray-300 font-bold uppercase">{gameMode === 'PVE' ? 'Bạn' : 'P1'}</div>
                      <div className={`w-4 h-4 rounded-full shadow-sm border border-gray-500 ${humanColor === PLAYER_BLACK ? 'stone-black' : 'stone-white'}`} title="Màu quân hiện tại"></div>
                  </div>
                  <div className="text-4xl font-bold text-white">{scores.p1}</div>
                  <div className="text-[10px] text-gray-500 mt-1">ĐIỂM</div>
                  {humanColor === currentPlayer && <div className="absolute bottom-0 left-0 w-full h-1 bg-yellow-500 animate-pulse"></div>}
              </div>

              <div className={`p-3 rounded-lg border relative overflow-hidden ${humanColor !== currentPlayer ? 'border-yellow-500 bg-yellow-900/20' : 'border-gray-700 bg-gray-800'}`}>
                  <div className="flex justify-between items-start mb-2">
                       <div className="text-xs text-gray-300 font-bold uppercase">{gameMode === 'PVE' ? 'Máy' : 'P2'}</div>
                       <div className={`w-4 h-4 rounded-full shadow-sm border border-gray-500 ${humanColor === PLAYER_BLACK ? 'stone-white' : 'stone-black'}`} title="Màu quân hiện tại"></div>
                  </div>
                  <div className="text-4xl font-bold text-white">{scores.p2}</div>
                  <div className="text-[10px] text-gray-500 mt-1">ĐIỂM</div>
                   {humanColor !== currentPlayer && <div className="absolute bottom-0 left-0 w-full h-1 bg-yellow-500 animate-pulse"></div>}
              </div>
          </div>

          {/* Game Info */}
           <div className="bg-gray-800 p-3 rounded-lg border border-gray-700">
              <div className="flex justify-between items-center mb-2">
                  <span className="text-xs uppercase text-gray-500 font-bold">Tổng lượt: {turnCount}</span>
                  <span className="text-xs text-blue-300">Đổi bên sau: {movesUntilNextSwap}</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-1.5 mb-3">
                 <div className="bg-blue-500 h-1.5 rounded-full transition-all duration-500" style={{ width: `${((turnCount % 30)/30)*100}%` }}></div>
              </div>
              
              <div className="text-center text-xs font-medium text-gray-400">
                Bạn đang cầm quân: <span className={`font-bold ${humanColor === PLAYER_BLACK ? 'text-gray-300' : 'text-white'}`}>{humanColor === PLAYER_BLACK ? "VOID (Đi trước)" : "LIGHT (Đi sau)"}</span>
              </div>

              {showSwapAlert && !isGameOver && (
                   <div className="mt-2 p-2 bg-yellow-900/50 border border-yellow-600 rounded text-xs text-yellow-400 text-center font-bold animate-bounce">
                     ⚠️ ĐÃ ĐỔI BÊN! KIỂM TRA MÀU CỜ!
                  </div>
              )}
           </div>

          {/* Controls */}
          <div className="flex flex-col gap-3">
            <div className="flex bg-gray-800 rounded-lg p-1">
              <button 
                onClick={() => { setGameMode('PVE'); resetGame(); }}
                className={`flex-1 py-2 rounded text-xs font-bold ${gameMode === 'PVE' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'}`}
              >
                Vs Máy
              </button>
              <button 
                onClick={() => { setGameMode('PVP'); resetGame(); }}
                className={`flex-1 py-2 rounded text-xs font-bold ${gameMode === 'PVP' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'}`}
              >
                2 Người
              </button>
            </div>
            {gameMode === 'PVE' && (
              <select 
                value={difficulty}
                onChange={(e) => { setDifficulty(e.target.value as Difficulty); resetGame(); }}
                className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-white"
              >
                <option value="EASY">Dễ (Random)</option>
                <option value="MEDIUM">Trung Bình</option>
                <option value="SUPER_STRONG">Siêu Mạnh (Local)</option>
                <option value="HARD">Cực Khó (Gemini AI)</option>
              </select>
            )}
          </div>

          {/* Rules */}
          <div className="bg-yellow-900/20 p-3 rounded border border-yellow-900/50 text-[10px] text-yellow-100/80 space-y-1">
            <p className="uppercase font-bold text-yellow-500">Cách chơi:</p>
            <ul className="list-disc pl-3 space-y-1">
                <li>Tạo hàng ≥ 5 -> <strong>ĐỐI THỦ nhận điểm = số quân</strong> & xóa cờ.</li>
                <li>Đổi màu cờ mỗi 30 lượt (icon ở bảng điểm).</li>
                <li>Hơn 100 điểm và đạt 200 điểm -> THẮNG LUÔN.</li>
                <li>Hòa điểm -> ai nhiều quân hơn bị TRỪ điểm.</li>
            </ul>
          </div>
          
          <button onClick={resetGame} className="mt-auto w-full bg-gray-700 py-3 rounded font-bold hover:bg-gray-600">Chơi Lại</button>
        </div>

        {/* Game Board Area */}
        <div className="w-full md:w-2/3 bg-[#eecfa1] p-2 md:p-8 flex items-center justify-center wood-texture relative">
          
          {/* Final Result Modal */}
          {isGameOver && finalResult && (
            <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/80 p-4">
                <div className="bg-white text-gray-900 p-8 rounded-2xl shadow-2xl text-center max-w-md w-full border-4 border-yellow-500">
                    <h2 className="text-3xl font-black mb-4 text-yellow-600">KẾT THÚC!</h2>
                    <div className="flex justify-center gap-8 mb-6 text-lg font-bold">
                        <div className="flex flex-col items-center">
                            <span>{gameMode === 'PVE' ? 'BẠN' : 'P1'}</span>
                            <span className="text-3xl">{scores.p1}</span>
                        </div>
                        <div className="flex flex-col items-center">
                            <span>{gameMode === 'PVE' ? 'MÁY' : 'P2'}</span>
                            <span className="text-3xl">{scores.p2}</span>
                        </div>
                    </div>
                    <div className="bg-gray-100 p-4 rounded-lg mb-6">
                        <div className="text-gray-500 text-sm uppercase tracking-wider font-bold mb-1">Người Thắng</div>
                        <div className="text-2xl font-black text-blue-600">{finalResult.winner}</div>
                        <div className="text-sm text-gray-600 mt-2 italic">{finalResult.reason}</div>
                    </div>
                    <button onClick={resetGame} className="w-full bg-blue-600 text-white py-3 rounded-lg font-bold shadow-lg hover:bg-blue-700 transition">
                        Ván Mới
                    </button>
                </div>
            </div>
          )}

          {/* Grid */}
          <div 
            className="grid gap-0 border-2 border-black bg-[#eecfa1] shadow-xl relative"
            style={{ 
              gridTemplateColumns: `repeat(${BOARD_SIZE}, minmax(0, 1fr))`,
              width: '100%',
              maxWidth: '550px',
              aspectRatio: '1/1'
            }}
          >
            {board.map((row, r) => (
              row.map((cell, c) => {
                const isRemoving = removedStones.some(s => s.r === r && s.c === c);
                const isLastMove = lastMove?.r === r && lastMove?.c === c;
                const isHovered = hoveredCell?.r === r && hoveredCell?.c === c;
                const canMove = cell === EMPTY && !isGameOver && !isAiThinking && (gameMode === 'PVP' || currentPlayer === humanColor);

                return (
                  <div 
                    key={`${r}-${c}`}
                    onClick={() => handleCellClick(r, c)}
                    onMouseEnter={() => canMove ? setHoveredCell({r, c}) : null}
                    onMouseLeave={() => setHoveredCell(null)}
                    className={`
                      relative border-[0.5px] border-black/20 
                      flex items-center justify-center
                      cursor-pointer hover:bg-black/10
                      ${isRemoving ? 'bg-red-500/50 z-10' : ''}
                    `}
                  >
                    {/* Ghost Stone */}
                    {isHovered && canMove && !isRemoving && (
                        <div 
                            className={`
                                absolute inset-0 m-auto
                                w-[60%] h-[60%] rounded-full opacity-50 box-border
                                ${currentPlayer === PLAYER_BLACK ? 'stone-black' : 'stone-white border-2 border-black/30'}
                            `}
                        ></div>
                    )}

                    {cell !== EMPTY && !isRemoving && (
                      <div 
                        className={`
                          absolute inset-0 m-auto
                          w-[80%] h-[80%] rounded-full stone animate-pop
                          ${cell === PLAYER_BLACK ? 'stone-black' : 'stone-white'}
                        `}
                      >
                        {isLastMove && (
                          <div className="absolute inset-0 m-auto w-2 h-2 bg-blue-500/50 rounded-full"></div>
                        )}
                      </div>
                    )}
                    {isRemoving && (
                        <div className="absolute inset-0 flex items-center justify-center text-white font-bold text-xs animate-ping z-20 drop-shadow-md">
                            +{removedStones.length}
                        </div>
                    )}
                  </div>
                );
              })
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

const root = createRoot(document.getElementById('root')!);
root.render(<App />);
