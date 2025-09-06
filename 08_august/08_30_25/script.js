// =============================
// Maze class
// =============================
class Maze {
    constructor(n) {
        this.n = n;
        this.grid = [];
        this.start = [n-1, 0];    // bottom-left
        this.end = [0, n-1];      // top-right
    }

    generate() {
        // Step 1: Initialize grid with all walkable cells
        this.grid = Array(this.n).fill(null).map(() => Array(this.n).fill(0));
        
        // Step 2: Create a guaranteed path using DFS
        this.createGuaranteedPath();
        
        // Step 3: Add random walls while maintaining solvability
        this.addRandomWalls();
        
        // Ensure start and end are always walkable
        this.grid[this.start[0]][this.start[1]] = 0;
        this.grid[this.end[0]][this.end[1]] = 0;
    }
    
    createGuaranteedPath() {
        // Simple path from start to end
        let [row, col] = this.start;
        const [endRow, endCol] = this.end;
        
        // Create a winding path to make it interesting
        while (row !== endRow || col !== endCol) {
            this.grid[row][col] = 0;
            
            // Randomly choose to move up or right (with some randomness)
            if (row > endRow && (col === endCol || Math.random() > 0.5)) {
                row--;
            } else if (col < endCol) {
                col++;
            } else if (row > endRow) {
                row--;
            }
        }
        this.grid[endRow][endCol] = 0;
    }
    
    addRandomWalls() {
        const wallProbability = 0.25;
        
        for (let i = 0; i < this.n; i++) {
            for (let j = 0; j < this.n; j++) {
                // Don't add walls on start, end, or guaranteed path
                if ((i === this.start[0] && j === this.start[1]) || 
                    (i === this.end[0] && j === this.end[1])) {
                    continue;
                }
                
                if (Math.random() < wallProbability) {
                    this.grid[i][j] = 1;
                    
                    // Check if maze is still solvable
                    if (!this.isSolvable()) {
                        this.grid[i][j] = 0; // Revert if not solvable
                    }
                }
            }
        }
    }
    
    isSolvable() {
        // Quick BFS to check if path exists
        const visited = Array(this.n).fill(null).map(() => Array(this.n).fill(false));
        const queue = [this.start];
        visited[this.start[0]][this.start[1]] = true;
        
        const directions = [[-1,0], [1,0], [0,-1], [0,1]];
        
        while (queue.length > 0) {
            const [row, col] = queue.shift();
            
            if (row === this.end[0] && col === this.end[1]) {
                return true;
            }
            
            for (const [dr, dc] of directions) {
                const newRow = row + dr;
                const newCol = col + dc;
                
                if (this.isValid(newRow, newCol) && 
                    !visited[newRow][newCol] && 
                    this.grid[newRow][newCol] === 0) {
                    visited[newRow][newCol] = true;
                    queue.push([newRow, newCol]);
                }
            }
        }
        
        return false;
    }
    
    isValid(row, col) {
        return row >= 0 && row < this.n && col >= 0 && col < this.n;
    }

    render(container, userPath = [], solutionPath = []) {
        container.innerHTML = '';
        container.style.gridTemplateColumns = `repeat(${this.n}, 25px)`;
        
        // Convert paths to string format for easy lookup
        const userPathSet = new Set(userPath.map(p => `${p[0]},${p[1]}`));
        const solutionPathSet = new Set(solutionPath.map(p => `${p[0]},${p[1]}`));
        
        for (let i = 0; i < this.n; i++) {
            for (let j = 0; j < this.n; j++) {
                const cell = document.createElement('div');
                cell.className = 'cell';
                cell.dataset.row = i;
                cell.dataset.col = j;
                
                // Base color
                if (this.grid[i][j] === 1) {
                    cell.classList.add('black');
                } else {
                    cell.classList.add('white');
                }
                
                // Special positions
                if (i === this.start[0] && j === this.start[1]) {
                    cell.classList.add('start');
                } else if (i === this.end[0] && j === this.end[1]) {
                    cell.classList.add('end');
                }
                
                // Path overlays
                const posKey = `${i},${j}`;
                if (solutionPathSet.has(posKey)) {
                    cell.classList.add('green');
                } else if (userPathSet.has(posKey)) {
                    cell.classList.add('blue');
                }
                
                container.appendChild(cell);
            }
        }
    }
}

// =============================
// PathFinder class using BFS (DP approach)
// =============================
class PathFinder {
    constructor(maze) {
        this.maze = maze;
    }

    shortestPath() {
        const n = this.maze.n;
        const grid = this.maze.grid;
        const start = this.maze.start;
        const end = this.maze.end;
        
        // BFS with parent tracking for path reconstruction
        const visited = Array(n).fill(null).map(() => Array(n).fill(false));
        const parent = Array(n).fill(null).map(() => Array(n).fill(null));
        const queue = [start];
        
        visited[start[0]][start[1]] = true;
        
        const directions = [[-1,0], [1,0], [0,-1], [0,1]];
        
        while (queue.length > 0) {
            const [row, col] = queue.shift();
            
            if (row === end[0] && col === end[1]) {
                // Found the end, reconstruct path
                return this.reconstructPath(parent, start, end);
            }
            
            for (const [dr, dc] of directions) {
                const newRow = row + dr;
                const newCol = col + dc;
                
                if (this.maze.isValid(newRow, newCol) && 
                    !visited[newRow][newCol] && 
                    grid[newRow][newCol] === 0) {
                    
                    visited[newRow][newCol] = true;
                    parent[newRow][newCol] = [row, col];
                    queue.push([newRow, newCol]);
                }
            }
        }
        
        return []; // No path found
    }
    
    reconstructPath(parent, start, end) {
        const path = [];
        let current = end;
        
        while (current) {
            path.unshift(current);
            current = parent[current[0]][current[1]];
        }
        
        return path;
    }
}

// =============================
// App class (controller)
// =============================
class App {
    constructor() {
        this.maze = null;
        this.userPath = [];
        this.solutionPath = [];
        this.isSelectingPath = false;
        this.bindUI();
    }

    bindUI() {
        document.getElementById('generateBtn')
            .addEventListener('click', () => this.handleGenerate());

        document.getElementById('showPathBtn')
            .addEventListener('click', () => this.handleShowPath());

        document.getElementById('clearPathBtn')
            .addEventListener('click', () => this.handleClearPath());

        document.getElementById('validateBtn')
            .addEventListener('click', () => this.handleValidate());
    }

    handleGenerate() {
        const n = parseInt(document.getElementById('sizeInput').value);
        if (n < 5 || n > 20) {
            this.showStatus('Please enter a size between 5 and 20.', 'error');
            return;
        }
        
        this.showStatus('Generating maze...', 'info');
        
        // Small delay to show loading message
        setTimeout(() => {
            this.maze = new Maze(n);
            this.maze.generate();
            this.userPath = [];
            this.solutionPath = [];
            
            this.maze.render(document.getElementById('grid'));
            this.enableUserClicks();
            
            // Enable buttons
            document.getElementById('showPathBtn').disabled = false;
            document.getElementById('clearPathBtn').disabled = false;
            document.getElementById('validateBtn').disabled = false;
            
            this.showStatus('Maze generated! Click tiles to create your path from S to E.', 'info');
        }, 100);
    }

    handleShowPath() {
        if (!this.maze) return;
        
        const finder = new PathFinder(this.maze);
        this.solutionPath = finder.shortestPath();
        
        if (this.solutionPath.length === 0) {
            this.showStatus('No path found! This shouldn\'t happen.', 'error');
            return;
        }
        
        this.maze.render(
            document.getElementById('grid'), 
            this.userPath, 
            this.solutionPath
        );
        this.enableUserClicks();
        
        this.showStatus(
            `Optimal path shown in green (${this.solutionPath.length} steps). ` +
            `Your path: ${this.userPath.length} steps.`, 
            'info'
        );
    }

    handleClearPath() {
        if (!this.maze) return;
        
        this.userPath = [];
        this.maze.render(
            document.getElementById('grid'), 
            this.userPath, 
            this.solutionPath
        );
        this.enableUserClicks();
        
        this.showStatus('Your path cleared. Try again!', 'info');
    }

    handleValidate() {
        if (!this.maze) {
            this.showStatus('Please generate a maze first.', 'error');
            return;
        }
        
        if (this.solutionPath.length === 0) {
            this.showStatus('Please show the optimal path first.', 'error');
            return;
        }
        
        if (this.userPath.length === 0) {
            this.showStatus('Please create a path by clicking on tiles.', 'error');
            return;
        }
        
        // Check if user path is valid (reaches the end)
        if (!this.isValidPath(this.userPath)) {
            this.showStatus('❌ Your path is invalid. Make sure it goes from start to end!', 'error');
            return;
        }
        
        if (this.userPath.length === this.solutionPath.length) {
            this.showStatus('🎉 Perfect! You found the optimal path!', 'success');
        } else if (this.userPath.length < this.solutionPath.length + 3) {
            this.showStatus('✅ Great job! Your path is very close to optimal.', 'success');
        } else {
            this.showStatus(
                `❌ Not optimal. Your path: ${this.userPath.length} steps, ` +
                `optimal: ${this.solutionPath.length} steps. Try again!`, 
                'error'
            );
        }
    }
    
    isValidPath(path) {
        if (path.length === 0) return false;
        
        const start = this.maze.start;
        const end = this.maze.end;
        
        // Check if path starts at start and ends at end
        const pathStart = path[0];
        const pathEnd = path[path.length - 1];
        
        if (pathStart[0] !== start[0] || pathStart[1] !== start[1]) return false;
        if (pathEnd[0] !== end[0] || pathEnd[1] !== end[1]) return false;
        
        // Check if each step is valid (adjacent and walkable)
        for (let i = 0; i < path.length; i++) {
            const [row, col] = path[i];
            
            // Check if cell is walkable
            if (this.maze.grid[row][col] === 1) return false;
            
            // Check if step is adjacent to previous (except first step)
            if (i > 0) {
                const [prevRow, prevCol] = path[i-1];
                const distance = Math.abs(row - prevRow) + Math.abs(col - prevCol);
                if (distance !== 1) return false;
            }
        }
        
        return true;
    }

    enableUserClicks() {
        const grid = document.getElementById('grid');
        const cells = grid.querySelectorAll('.cell');
        
        cells.forEach(cell => {
            cell.onclick = (e) => this.handleCellClick(e);
        });
    }
    
    handleCellClick(e) {
        const row = parseInt(e.target.dataset.row);
        const col = parseInt(e.target.dataset.col);
        
        // Don't allow clicking on walls
        if (this.maze.grid[row][col] === 1) {
            this.showStatus('Cannot step on walls!', 'error');
            return;
        }
        
        // If this is the first click, it must be the start
        if (this.userPath.length === 0) {
            if (row !== this.maze.start[0] || col !== this.maze.start[1]) {
                this.showStatus('You must start from the red start tile (S)!', 'error');
                return;
            }
        } else {
            // Check if click is adjacent to last position
            const lastPos = this.userPath[this.userPath.length - 1];
            const distance = Math.abs(row - lastPos[0]) + Math.abs(col - lastPos[1]);
            
            if (distance !== 1) {
                this.showStatus('You can only move to adjacent tiles!', 'error');
                return;
            }
        }
        
        // Add to path
        this.userPath.push([row, col]);
        
        // Re-render
        this.maze.render(
            document.getElementById('grid'), 
            this.userPath, 
            this.solutionPath
        );
        this.enableUserClicks();
        
        // Check if reached end
        if (row === this.maze.end[0] && col === this.maze.end[1]) {
            this.showStatus(
                `🎯 You reached the end in ${this.userPath.length} steps! ` +
                `Now click "Validate My Path" to see how you did.`, 
                'success'
            );
        } else {
            this.showStatus(
                `Path length: ${this.userPath.length} steps. Keep going to reach the gold end tile (E)!`, 
                'info'
            );
        }
    }
    
    showStatus(message, type = 'info') {
        const status = document.getElementById('status');
        status.textContent = message;
        status.className = type;
    }
}

// =============================
// Entry point
// =============================
document.addEventListener('DOMContentLoaded', () => {
    new App();
});