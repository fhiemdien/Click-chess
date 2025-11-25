import React, { useState, useEffect, useRef, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import Peer, { DataConnection } from 'peerjs';

// --- Constants ---
const BOARD_SIZE = 15;
const PLAYER_BLACK = 1; // Stone Color Black (VOID)
const PLAYER_WHITE = 2; // Stone Color White (LIGHT)
const EMPTY = 0;

type CellValue = 0 | 1 | 2;
type BoardState = CellValue[][];
type Difficulty = 'EASY' | 'MEDIUM' | 'SUPER_STRONG';
type GameMode = 'PVP' | 'PVE' | 'ONLINE';

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

    // const opponent = aiPlayer === PLAYER_BLACK ? PLAYER_WHITE : PLAYER_BLACK;
    
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
    return candidates[0]?.move || { r: 7, c: 7 };
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

  // --- Online Mode States ---
  const [peer, setPeer] = useState<Peer | null>(null);
  const [conn, setConn] = useState<DataConnection | null>(null);
  const [myPeerId, setMyPeerId] = useState<string>('');
  const [connectToId, setConnectToId] = useState<string>('');
  const [onlineStatus, setOnlineStatus] = useState<'IDLE' | 'WAITING' | 'CONNECTED'>('IDLE');
  const [onlinePlayerColor, setOnlinePlayerColor] = useState<CellValue>(PLAYER_BLACK); // My color in Online mode

  const movesUntilNextSwap = 30 - (turnCount % 30);

  const playSound = (type: 'click' | 'clear') => {};

  useEffect(() => {
    return () => {
      // Cleanup peer on unmount
      if (peer) peer.destroy();
    };
  }, [peer]);

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

  const initOnlineGame = (isHost: boolean) => {
    const newPeer = new Peer();
    setPeer(newPeer);
    setOnlineStatus('WAITING');
    resetGame();
    setGameMode('ONLINE');

    newPeer.on('open', (id) => {
      setMyPeerId(id);
      if (isHost) {
          // Host is Player 1 (Black/Void)
          setOnlinePlayerColor(PLAYER_BLACK);
          setHumanColor(PLAYER_BLACK);
      } else {
          // Guest logic handles connection below
      }
    });

    newPeer.on('connection', (connection) => {
      // Logic for HOST receiving connection
      setConn(connection);
      setOnlineStatus('CONNECTED');
      setupConnectionHandlers(connection);
    });
  };

  const joinOnlineGame = () => {
    if (!connectToId || !peer) return;
    const connection = peer.connect(connectToId);
    setConn(connection);
    // Guest is Player 2 (White/Light)
    setOnlinePlayerColor(PLAYER_WHITE);
    setHumanColor(PLAYER_WHITE); 
    setupConnectionHandlers(connection);
  };

  const setupConnectionHandlers = (connection: DataConnection) => {
      connection.on('open', () => {
          setOnlineStatus('CONNECTED');
      });
      connection.on('data', (data: any) => {
          if (data && data.type === 'MOVE') {
              // Received move from opponent
              handleRemoteMove(data.r, data.c);
          } else if (data && data.type === 'RESET') {
             resetGame(); // Allow opponent to trigger reset
          }
      });
      connection.on('close', () => {
          alert('Đối thủ đã thoát!');
          setOnlineStatus('IDLE');
          setGameMode('PVE');
      });
      connection.on('error', (err) => {
          console.error(err);
          alert('Lỗi kết nối!');
      });
  };

  const handleRemoteMove = (r: number, c: number) => {
      // Need access to latest state, so we use functional updates in processTurn if needed
      // But simpler here: get latest board state via ref or relying on closure if component updates
      // React state closure is tricky. For simplicity, we use the `board` from state but we need to be careful.
      // Actually, since this is an event handler, it might see stale state.
      // Best way: Use a ref for the board or simply pass the updater to setBoard.
      // HOWEVER, `processTurn` needs the CURRENT board.
      
      // Let's use a workaround: dispatch an event or use a ref. 
      // For this specific app structure, let's trust React's rerender for now, 
      // but note that 'data' listener needs to be refreshed or access refs.
      // To fix stale closures in `setupConnectionHandlers`, we should use a ref for the handlers or `useEffect` dependency.
      
      // FIX: trigger a state update that handles the move
      setLastRemoteMove({r, c});
  };

  // Helper to trigger remote move processing when state updates
  const [lastRemoteMove, setLastRemoteMove] = useState<{r:number, c:number} | null>(null);
  useEffect(() => {
      if (lastRemoteMove) {
          const { r, c } = lastRemoteMove;
          // Verify it's a valid move (it should be)
          if (board[r][c] === EMPTY) {
               const newBoard = board.map(row => [...row]);
               newBoard[r][c] = currentPlayer;
               setBoard(newBoard);
               setLastMove({ r, c });
               playSound('click');
               processTurn(newBoard, r, c, currentPlayer, humanColor, scores, turnCount);
          }
          setLastRemoteMove(null);
      }
  }, [lastRemoteMove]);


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
       // Logic for point attribution:
       // If I moved, and it exploded, OPPONENT gets points.
       // Who is 'I' in PVE/PVP?
       // In PVE: isP1Move = playerWhoMoved === currentHumanColor
       // In ONLINE: P1 is Host, P2 is Guest. 
       // We track P1/P2 scores directly. 
       // If PLAYER_BLACK moved -> P1 moved. If PLAYER_WHITE moved -> P2 moved.
       
       // BUT, due to swapping, Player 1 might be holding White.
       // Let's rely on who made the move.
       // We need to know who is 'P1' and who is 'P2' conceptually.
       // In this game logic:
       // P1 starts with Black.
       // P2 starts with White.
       // When swapped: P1 holds White, P2 holds Black.
       
       // Score logic update:
       // If the player who moved is currently P1 -> P2 gets points.
       // If the player who moved is currently P2 -> P1 gets points.
       
       // Check if the current player (who moved) is P1 or P2.
       // Logic: 
       // Initial: P1=Black, P2=White.
       // Swap 1 (30 moves): P1=White, P2=Black.
       // Swap 2 (60 moves): P1=Black, P2=White.
       // So: if (turnCount / 30) is even (0, 2..), P1 is Black.
       // If (turnCount / 30) is odd (1, 3..), P1 is White.
       
       const cycle = Math.floor(currentTurnBase / 30); // Use PREVIOUS turn count for current state
       const p1IsBlack = cycle % 2 === 0;
       
       let isP1 = false;
       if (p1IsBlack) {
           if (playerWhoMoved === PLAYER_BLACK) isP1 = true;
       } else {
           if (playerWhoMoved === PLAYER_WHITE) isP1 = true;
       }
       
       if (isP1) {
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

     const p1Name = gameMode === 'PVE' ? "BẠN" : (gameMode === 'ONLINE' ? "P1 (HOST)" : "NGƯỜI CHƠI 1");
     const p2Name = gameMode === 'PVE' ? "MÁY" : (gameMode === 'ONLINE' ? "P2 (GUEST)" : "NGƯỜI CHƠI 2");

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
    
    // Permission checks
    if (gameMode === 'PVE' && currentPlayer !== humanColor) return;
    if (gameMode === 'ONLINE') {
        // Can only move if it's my turn AND the current player matches my assigned color
        // Note: humanColor updates on Swap. onlinePlayerColor is static (Host=P1, Guest=P2).
        
        // We need to check if 'I' am the one who should be moving.
        // humanColor tracks who 'I' am currently playing as (Black/White).
        // currentPlayer is whose turn it is (Black/White).
        // So:
        if (currentPlayer !== humanColor) return;
    }

    if (board[r][c] !== EMPTY) return;

    const newBoard = board.map(row => [...row]);
    newBoard[r][c] = currentPlayer;
    
    setBoard(newBoard);
    setLastMove({ r, c });
    playSound('click');

    // Send move if online
    if (gameMode === 'ONLINE' && conn) {
        conn.send({ type: 'MOVE', r, c });
    }

    processTurn(newBoard, r, c, currentPlayer, humanColor, scores, turnCount);
  };

  const makeAiMove = async (currentBoard: BoardState, aiPlayer: CellValue, currentHumanColor: CellValue, aiScore: number, humanScore: number, currentTurnCount: number) => {
    let move: { r: number, c: number } | null = null;

    if (difficulty === 'EASY') {
      move = getEasyMove(currentBoard, aiPlayer);
    } else if (difficulty === 'MEDIUM') {
      await new Promise(resolve => setTimeout(resolve, 300));
      move = getMediumMove(currentBoard, aiPlayer);
    } else {
      // Super Strong
      await new Promise(resolve => setTimeout(resolve, 500));
      move = getSuperStrongMove(currentBoard, aiPlayer);
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
                      <div className="text-xs text-gray-300 font-bold uppercase">
                        {gameMode === 'PVE' ? 'Bạn' : (gameMode === 'ONLINE' ? 'P1 (Host)' : 'P1')}
                      </div>
                      <div className={`w-4 h-4 rounded-full shadow-sm border border-gray-500 ${humanColor === PLAYER_BLACK ? 'stone-black' : 'stone-white'}`} title="Màu quân hiện tại"></div>
                  </div>
                  <div className="text-4xl font-bold text-white">{scores.p1}</div>
                  <div className="text-[10px] text-gray-500 mt-1">ĐIỂM</div>
                  {humanColor === currentPlayer && <div className="absolute bottom-0 left-0 w-full h-1 bg-yellow-500 animate-pulse"></div>}
              </div>

              <div className={`p-3 rounded-lg border relative overflow-hidden ${humanColor !== currentPlayer ? 'border-yellow-500 bg-yellow-900/20' : 'border-gray-700 bg-gray-800'}`}>
                  <div className="flex justify-between items-start mb-2">
                       <div className="text-xs text-gray-300 font-bold uppercase">
                         {gameMode === 'PVE' ? 'Máy' : (gameMode === 'ONLINE' ? 'P2 (Guest)' : 'P2')}
                       </div>
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
            {/* Mode Select */}
            <div className="flex bg-gray-800 rounded-lg p-1">
              <button 
                onClick={() => { setGameMode('PVE'); resetGame(); if(peer) peer.destroy(); setPeer(null); setOnlineStatus('IDLE'); }}
                className={`flex-1 py-2 rounded text-xs font-bold ${gameMode === 'PVE' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'}`}
              >
                Vs Máy
              </button>
              <button 
                onClick={() => { setGameMode('PVP'); resetGame(); if(peer) peer.destroy(); setPeer(null); setOnlineStatus('IDLE'); }}
                className={`flex-1 py-2 rounded text-xs font-bold ${gameMode === 'PVP' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'}`}
              >
                2 Người
              </button>
               <button 
                onClick={() => { setGameMode('ONLINE'); setOnlineStatus('IDLE'); }}
                className={`flex-1 py-2 rounded text-xs font-bold ${gameMode === 'ONLINE' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'}`}
              >
                Online
              </button>
            </div>

            {/* PVE Settings */}
            {gameMode === 'PVE' && (
              <select 
                value={difficulty}
                onChange={(e) => { setDifficulty(e.target.value as Difficulty); resetGame(); }}
                className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-white"
              >
                <option value="EASY">Dễ (Random)</option>
                <option value="MEDIUM">Trung Bình</option>
                <option value="SUPER_STRONG">Siêu Mạnh (Local)</option>
              </select>
            )}

             {/* Online Settings */}
            {gameMode === 'ONLINE' && (
                <div className="bg-gray-800 p-3 rounded-lg border border-gray-700 space-y-3">
                   {onlineStatus === 'IDLE' && (
                       <div className="flex gap-2">
                           <button onClick={() => initOnlineGame(true)} className="flex-1 bg-green-600 hover:bg-green-700 text-white text-xs py-2 rounded font-bold">Tạo Phòng</button>
                           <div className="flex-1 flex gap-1">
                                <input 
                                    type="text" 
                                    placeholder="Nhập ID"
                                    value={connectToId}
                                    onChange={(e) => setConnectToId(e.target.value)}
                                    className="w-full bg-gray-900 border border-gray-600 rounded px-2 text-xs text-white"
                                />
                                <button onClick={() => { const p = new Peer(); setPeer(p); p.on('open', ()=>joinOnlineGame()); }} className="bg-blue-600 px-2 rounded text-white text-xs font-bold">Vào</button>
                           </div>
                       </div>
                   )}

                   {onlineStatus === 'WAITING' && (
                       <div className="text-center">
                           <p className="text-xs text-gray-400 mb-1">Mã phòng của bạn:</p>
                           <div className="flex gap-2 items-center justify-center bg-black/30 p-2 rounded mb-2">
                               <code className="text-yellow-400 font-mono text-sm">{myPeerId}</code>
                               <button 
                                onClick={() => { navigator.clipboard.writeText(window.location.origin + '?join=' + myPeerId); alert('Đã copy link! Gửi cho bạn bè nhé.'); }}
                                className="text-xs bg-gray-700 hover:bg-gray-600 px-2 py-1 rounded text-white"
                               >
                                   Copy Link
                               </button>
                           </div>
                           <p className="text-xs text-gray-500 animate-pulse">Đang đợi đối thủ...</p>
                       </div>
                   )}

                   {onlineStatus === 'CONNECTED' && (
                       <div className="bg-green-900/30 border border-green-800 p-2 rounded text-center">
                           <p className="text-green-400 text-xs font-bold">🟢 Đã kết nối!</p>
                           <p className="text-[10px] text-gray-400 mt-1">Bạn là: {onlinePlayerColor === PLAYER_BLACK ? 'P1 (Host)' : 'P2 (Guest)'}</p>
                       </div>
                   )}
                </div>
            )}
          </div>

          {/* Rules */}
          <div className="bg-yellow-900/20 p-3 rounded border border-yellow-900/50 text-[10px] text-yellow-100/80 space-y-1">
            <p className="uppercase font-bold text-yellow-500">Cách chơi:</p>
            <ul className="list-disc pl-3 space-y-1">
                <li>Tạo hàng &ge; 5 &rarr; <strong>ĐỐI THỦ nhận điểm = số quân</strong> và xóa cờ.</li>
                <li>Đổi màu cờ mỗi 30 lượt (icon ở bảng điểm).</li>
                <li>Hơn 100 điểm và đạt 200 điểm &rarr; THẮNG LUÔN.</li>
                <li>Hòa điểm &rarr; ai nhiều quân hơn bị TRỪ điểm.</li>
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
                
                // Allow move if:
                // 1. Cell is empty
                // 2. Not game over
                // 3. Not AI thinking
                // 4. PVP OR (PVE and my turn) OR (ONLINE and my turn AND I am allowed to move)
                let canMove = cell === EMPTY && !isGameOver && !isAiThinking;
                
                if (canMove) {
                    if (gameMode === 'PVE') {
                        if (currentPlayer !== humanColor) canMove = false;
                    } else if (gameMode === 'ONLINE') {
                        if (currentPlayer !== humanColor) canMove = false;
                    }
                }

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

// Check for join link parameter on load
const urlParams = new URLSearchParams(window.location.search);
const joinId = urlParams.get('join');
// We can auto-populate the join ID if we wanted to, but for now just letting user know
if(joinId) {
    // Logic to handle auto join could go here, for now simpler to just let user copy paste or manual entry
    // Or set initial state for input.
    // We will do that in component via a prop or ref if needed, but manual entry is safer for this demo.
}

const root = createRoot(document.getElementById('root')!);
root.render(<App />);