import numpy as np
import matplotlib.pyplot as plt
import heapq
from typing import List, Tuple, Dict, Optional

def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Diagonal distance heuristic for 8-directional grid movement."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)  # Octile distance

def get_neighbors(node: Tuple[int, int], grid: np.ndarray) -> List[Tuple[int, int]]:
    """Return valid 8-directional neighbors (including diagonals) that are free (0)."""
    neighbors = []
    x, y = node
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue                   # Skip current cell
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                                           # Check for diagonal movement through blocked corners
                if dx != 0 and dy != 0:
                    if grid[x + dx, y] == 1 or grid[x, y + dy] == 1:
                        continue           # Block diagonal movement through obstacles
                if grid[nx, ny] == 0:
                    neighbors.append((nx, ny))
    return neighbors

def astar(start: Tuple[int, int], goal: Tuple[int, int], grid: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    """Optimized A* with priority updates and path reconstruction."""
    open_set = []
    heapq.heappush(open_set, (0, 0, start))     # (priority, cost, node)

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    cost_so_far: Dict[Tuple[int, int], float] = {start: 0}

    while open_set:
        _, current_cost, current = heapq.heappop(open_set)

        if current == goal:
            break

        for neighbor in get_neighbors(current, grid):
            move_cost = np.sqrt((neighbor[0] - current[0])**2 + (neighbor[1] - current[1])**2)
            new_cost = current_cost + move_cost

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                heapq.heappush(open_set, (priority, new_cost, neighbor))
                came_from[neighbor] = current

    if goal not in came_from:
        return None

    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()

    return path

def create_parking_lot(rows=10, cols=10) -> np.ndarray:
    grid = np.zeros((rows, cols), dtype=int)

    grid[3:7, 4:6] = 1  # Central lane
    for i in [1, 8]:
        grid[i, 2:8] = 0  # Parking rows
        grid[i, 2:8:2] = 1  # Barriers between spots


    grid[0, 0] = 0  # Entrance
    grid[-1, -1] = 0  # VIP parking spot
    return grid

def plot_parking(grid: np.ndarray, path: List[Tuple[int, int]] = None) -> None:
    plt.figure(figsize=(8, 8))

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 1:
                plt.fill_between([j, j+1], i, i+1, color='#2c3e50', edgecolor='k')  # Obstacles
            elif (i, j) == (0, 0):
                plt.fill_between([j, j+1], i, i+1, color='#3498db', alpha=0.3)  # Start zone
            elif (i, j) == (grid.shape[0]-1, grid.shape[1]-1):
                plt.fill_between([j, j+1], i, i+1, color='#e74c3c', alpha=0.3)  # VIP spot

    if path:
        path_x = [p[1] + 0.5 for p in path]
        path_y = [p[0] + 0.5 for p in path]
        plt.plot(path_x, path_y, color='#27ae60', linewidth=3, linestyle='--', marker='o', markersize=8)

    plt.gca().invert_yaxis()
    plt.xticks(np.arange(0, cols+1))
    plt.yticks(np.arange(0, rows+1))
    plt.grid(color='#95a5a6', linestyle='--', linewidth=0.5)
    plt.title(" Vehicle Autonomous Parking System", pad=20)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Initialize parking environment

    rows, cols = 10, 10                                   # CHANGE THIS TO GET DIFFERENT RESULTS.
    grid = create_parking_lot(rows, cols)
    start = (0, 0)
    goal = (rows-1, cols-1)                               # THIS IS YOUR PARK LOCATION

    path = astar(start, goal, grid)

    if path:
        print(f"Found optimal path with {len(path)} steps:")
        print(np.array(path))
        plot_parking(grid, path)
    else:
        print("No valid path found. Please reconfigure parking lot.")
