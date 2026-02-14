"""
Grid Environment for Path Planning
Supports static and dynamic obstacles
"""

import numpy as np
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class GridEnvironment:
    """
    2D grid environment with obstacle support
    """
    
    def __init__(self, width: int, height: int, obstacle_density: float = 0.2):
        """
        Initialize grid environment
        
        Args:
            width: Grid width in cells
            height: Grid height in cells
            obstacle_density: Fraction of cells occupied by static obstacles
        """
        self.width = width
        self.height = height
        self.obstacle_density = obstacle_density
        
        # Grid: 0 = free, 1 = static obstacle, 2 = dynamic obstacle
        self.grid = np.zeros((width, height), dtype=int)
        
        # Generate static obstacles
        self._generate_obstacles()
        
        # Dynamic obstacles
        self.dynamic_obstacles = []
    
    def _generate_obstacles(self) -> None:
        """Generate random static obstacles"""
        n_obstacles = int(self.width * self.height * self.obstacle_density)
        
        for _ in range(n_obstacles):
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            self.grid[x, y] = 1
    
    def add_dynamic_obstacle(self, obs_id: int, px: float, py: float, 
                           vx: float = 0.0, vy: float = 0.0) -> None:
        """
        Add dynamic obstacle
        
        Args:
            obs_id: Unique obstacle identifier
            px, py: Initial position
            vx, vy: Initial velocity
        """
        obstacle = {
            'id': obs_id,
            'px': px,
            'py': py,
            'vx': vx,
            'vy': vy
        }
        self.dynamic_obstacles.append(obstacle)
    
    def update_dynamic_obstacles(self, dt: float = 0.1) -> None:
        """
        Update dynamic obstacle positions
        
        Args:
            dt: Time step in seconds
        """
        for obs in self.dynamic_obstacles:
            # Constant velocity motion
            obs['px'] += obs['vx'] * dt
            obs['py'] += obs['vy'] * dt
            
            # Add small random perturbation (process noise)
            obs['px'] += np.random.normal(0, 0.05)
            obs['py'] += np.random.normal(0, 0.05)
            
            # Bounce off boundaries
            if obs['px'] < 0 or obs['px'] >= self.width:
                obs['vx'] *= -1
                obs['px'] = np.clip(obs['px'], 0, self.width - 1)
            
            if obs['py'] < 0 or obs['py'] >= self.height:
                obs['vy'] *= -1
                obs['py'] = np.clip(obs['py'], 0, self.height - 1)
    
    def is_valid(self, pos: Tuple[int, int]) -> bool:
        """
        Check if position is valid (in bounds and not static obstacle)
        
        Args:
            pos: (x, y) position
            
        Returns:
            True if valid, False otherwise
        """
        x, y = pos
        
        # Bounds check
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        
        # Static obstacle check
        if self.grid[x, y] == 1:
            return False
        
        return True
    
    def is_collision_free(self, pos: Tuple[float, float], 
                         safety_margin: float = 1.0) -> bool:
        """
        Check if position has sufficient clearance from obstacles
        
        Args:
            pos: (x, y) position
            safety_margin: Minimum distance from obstacles
            
        Returns:
            True if collision-free, False otherwise
        """
        if not self.is_valid((int(pos[0]), int(pos[1]))):
            return False
        
        # Check distance to dynamic obstacles
        for obs in self.dynamic_obstacles:
            dist = np.sqrt((pos[0] - obs['px'])**2 + (pos[1] - obs['py'])**2)
            if dist < safety_margin:
                return False
        
        return True
    
    def get_random_free_position(self) -> Tuple[int, int]:
        """Get random free position in grid"""
        max_attempts = 1000
        for _ in range(max_attempts):
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            if self.is_valid((x, y)):
                return (x, y)
        
        raise ValueError("Could not find free position")
    
    def visualize(self, path: List[Tuple] = None, 
                 start: Tuple = None, goal: Tuple = None,
                 penalty_map: Dict = None,
                 save_path: str = None) -> None:
        """
        Visualize environment with optional path
        
        Args:
            path: List of path nodes
            start: Start position
            goal: Goal position
            penalty_map: Dictionary of penalty costs
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw grid
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Draw static obstacles
        for i in range(self.width):
            for j in range(self.height):
                if self.grid[i, j] == 1:
                    rect = patches.Rectangle((i, j), 1, 1, 
                                            linewidth=0, 
                                            facecolor='black', 
                                            alpha=0.8)
                    ax.add_patch(rect)
        
        # Draw penalty map as heatmap
        if penalty_map:
            penalty_grid = np.zeros((self.width, self.height))
            for (x, y), cost in penalty_map.items():
                if 0 <= x < self.width and 0 <= y < self.height:
                    penalty_grid[x, y] = cost
            
            if np.max(penalty_grid) > 0:
                ax.imshow(penalty_grid.T, origin='lower', 
                         cmap='Reds', alpha=0.3, 
                         extent=[0, self.width, 0, self.height])
        
        # Draw dynamic obstacles
        for obs in self.dynamic_obstacles:
            circle = plt.Circle((obs['px'], obs['py']), 0.5, 
                              color='red', alpha=0.6)
            ax.add_patch(circle)
            
            # Draw velocity vector
            ax.arrow(obs['px'], obs['py'], 
                    obs['vx'], obs['vy'],
                    head_width=0.3, head_length=0.2, 
                    fc='darkred', ec='darkred', alpha=0.6)
        
        # Draw path
        if path and len(path) > 1:
            path_x = [p[0] + 0.5 for p in path]
            path_y = [p[1] + 0.5 for p in path]
            ax.plot(path_x, path_y, 'b-', linewidth=2, label='Path')
        
        # Draw start
        if start:
            ax.plot(start[0] + 0.5, start[1] + 0.5, 'go', 
                   markersize=15, label='Start')
        
        # Draw goal
        if goal:
            ax.plot(goal[0] + 0.5, goal[1] + 0.5, 'r*', 
                   markersize=20, label='Goal')
        
        ax.legend(loc='upper right')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Grid Environment ({self.width}x{self.height})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def get_active_region(self, path: List[Tuple], 
                         expansion: float = 3.0) -> List[Tuple]:
        """
        Get active region around current path for efficient penalty updates
        
        Args:
            path: Current planned path
            expansion: Multiple of path length to expand region
            
        Returns:
            List of (x, y) nodes in active region
        """
        if not path:
            return []
        
        # Calculate path length
        path_length = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            path_length += np.sqrt(dx**2 + dy**2)
        
        # Active distance
        d_active = expansion * path_length
        
        # Find nodes within distance
        active_nodes = []
        for node in path:
            for dx in range(-int(d_active), int(d_active) + 1):
                for dy in range(-int(d_active), int(d_active) + 1):
                    x = node[0] + dx
                    y = node[1] + dy
                    
                    if self.is_valid((x, y)):
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist <= d_active:
                            active_nodes.append((x, y))
        
        # Remove duplicates
        active_nodes = list(set(active_nodes))
        
        return active_nodes
