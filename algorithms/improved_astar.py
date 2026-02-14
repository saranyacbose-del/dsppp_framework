"""
Improved A* Algorithm with Dynamic Penalty Costs
Based on the DSPPP framework
"""

import numpy as np
import heapq
from typing import List, Tuple, Dict, Set, Optional
import time


class ImprovedAStar:
    """
    Improved A* algorithm integrating:
    - Dynamic penalty costs for obstacle avoidance
    - Semantic awareness for context-based routing
    - Path smoothness optimization
    - Real-time replanning capability
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the Improved A* planner
        
        Args:
            config: Configuration dictionary with parameters
        """
        self.lambda_base = config.get('lambda_base', 5.0)
        self.sigma = config.get('sigma', 2.0)
        self.delta_t = config.get('delta_t', 1.0)
        self.gamma = config.get('gamma', 0.15)
        self.beta = config.get('beta', 0.1)
        self.w_base = config.get('w_base', 1.0)
        self.t_max = config.get('t_max', 5.0)
        
        self.open_set = []
        self.closed_set = set()
        self.g_score = {}
        self.f_score = {}
        self.parent = {}
        
    def search(self, start: Tuple, goal: Tuple, graph, 
               obstacles: List = None, 
               penalty_map: Dict = None,
               semantic_costs: Dict = None) -> Tuple[List, Dict]:
        """
        Execute A* search with integrated costs
        
        Args:
            start: Start node coordinates
            goal: Goal node coordinates
            graph: NetworkX graph or grid environment
            obstacles: List of dynamic obstacles
            penalty_map: Dynamic penalty cost map
            semantic_costs: Semantic cost dictionary
            
        Returns:
            path: List of nodes from start to goal
            stats: Dictionary of search statistics
        """
        t_start = time.perf_counter()
        
        # Initialize
        self.open_set = []
        self.closed_set = set()
        self.g_score = {start: 0}
        self.f_score = {start: self.heuristic(start, goal, graph, obstacles)}
        self.parent = {start: None}
        
        heapq.heappush(self.open_set, (self.f_score[start], start))
        
        nodes_explored = 0
        
        while self.open_set:
            # Check time budget
            if (time.perf_counter() - t_start) > self.t_max:
                return self._best_partial_path(goal), {
                    'success': False,
                    'time': time.perf_counter() - t_start,
                    'nodes_explored': nodes_explored,
                    'reason': 'timeout'
                }
            
            # Get node with minimum f_score
            _, current = heapq.heappop(self.open_set)
            
            if current in self.closed_set:
                continue
                
            nodes_explored += 1
            
            # Goal check
            if current == goal:
                path = self._reconstruct_path(current)
                return path, {
                    'success': True,
                    'time': time.perf_counter() - t_start,
                    'nodes_explored': nodes_explored,
                    'path_length': self._path_length(path),
                    'path_cost': self.g_score[current]
                }
            
            self.closed_set.add(current)
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current, graph):
                if neighbor in self.closed_set:
                    continue
                
                # Calculate total cost
                edge_cost = self._total_cost(
                    current, neighbor, graph, 
                    obstacles, penalty_map, semantic_costs
                )
                
                tentative_g = self.g_score[current] + edge_cost
                
                if neighbor not in self.g_score or tentative_g < self.g_score[neighbor]:
                    self.parent[neighbor] = current
                    self.g_score[neighbor] = tentative_g
                    self.f_score[neighbor] = tentative_g + self.heuristic(
                        neighbor, goal, graph, obstacles
                    )
                    heapq.heappush(self.open_set, (self.f_score[neighbor], neighbor))
        
        # No path found
        return [], {
            'success': False,
            'time': time.perf_counter() - t_start,
            'nodes_explored': nodes_explored,
            'reason': 'no_path'
        }
    
    def _total_cost(self, ni: Tuple, nj: Tuple, graph,
                   obstacles: List, penalty_map: Dict, 
                   semantic_costs: Dict) -> float:
        """
        Calculate comprehensive cost (Equation 14 from paper)
        
        C_total = C_base + C_penalty + C_semantic + C_smooth
        """
        # Base movement cost
        c_base = self._base_cost(ni, nj, graph)
        
        # Dynamic penalty cost
        c_penalty = self._penalty_cost(nj, obstacles, penalty_map)
        
        # Semantic cost
        c_semantic = self._semantic_cost(ni, nj, semantic_costs)
        
        # Smoothness cost
        c_smooth = self._smoothness_cost(ni, nj)
        
        return c_base + c_penalty + c_semantic + c_smooth
    
    def _base_cost(self, ni: Tuple, nj: Tuple, graph) -> float:
        """Euclidean distance or edge weight"""
        if hasattr(graph, 'has_edge') and graph.has_edge(ni, nj):
            # NetworkX graph
            return graph[ni][nj].get('weight', 1.0)
        else:
            # Grid environment - Euclidean distance
            return np.sqrt((nj[0] - ni[0])**2 + (nj[1] - ni[1])**2)
    
    def _penalty_cost(self, nj: Tuple, obstacles: List, 
                     penalty_map: Dict) -> float:
        """
        Dynamic penalty cost (Equation 17)
        Uses precomputed penalty map for efficiency
        """
        if penalty_map is None or obstacles is None:
            return 0.0
        
        return penalty_map.get(nj, 0.0)
    
    def _semantic_cost(self, ni: Tuple, nj: Tuple, 
                       semantic_costs: Dict) -> float:
        """
        Semantic cost based on user preferences (Equation 19)
        """
        if semantic_costs is None:
            return 0.0
        
        edge = (ni, nj)
        return semantic_costs.get(edge, 0.0)
    
    def _smoothness_cost(self, ni: Tuple, nj: Tuple) -> float:
        """
        Path smoothness cost (Equation 20)
        Penalizes sharp turns
        """
        if self.parent.get(ni) is None:
            return 0.0
        
        prev = self.parent[ni]
        
        # Calculate heading angles
        theta_prev = np.arctan2(ni[1] - prev[1], ni[0] - prev[0])
        theta_curr = np.arctan2(nj[1] - ni[1], nj[0] - ni[0])
        
        # Angular deviation
        angle_diff = abs(theta_curr - theta_prev)
        
        # Normalize to [0, pi]
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff
        
        return self.beta * angle_diff
    
    def heuristic(self, n: Tuple, goal: Tuple, graph, 
                  obstacles: List = None) -> float:
        """
        Enhanced heuristic with obstacle density awareness (Equation 22)
        
        h(n, g) = h_euclidean(n, g) * (1 + γ * ρ(n))
        """
        # Euclidean distance to goal
        h_euclidean = np.sqrt((goal[0] - n[0])**2 + (goal[1] - n[1])**2)
        
        # Obstacle density factor
        if obstacles and self.gamma > 0:
            density = self._obstacle_density(n, obstacles)
            return h_euclidean * (1 + self.gamma * density)
        
        return h_euclidean
    
    def _obstacle_density(self, n: Tuple, obstacles: List, 
                         r_sense: float = 5.0) -> float:
        """
        Calculate local obstacle density (Equation 24)
        """
        count = 0
        for obs in obstacles:
            obs_pos = (obs['px'], obs['py'])
            dist = np.sqrt((n[0] - obs_pos[0])**2 + (n[1] - obs_pos[1])**2)
            if dist <= r_sense:
                count += 1
        
        area = np.pi * r_sense**2
        return count / area if area > 0 else 0.0
    
    def _get_neighbors(self, node: Tuple, graph) -> List[Tuple]:
        """Get neighboring nodes (8-connected for grid)"""
        if hasattr(graph, 'neighbors'):
            # NetworkX graph
            return list(graph.neighbors(node))
        else:
            # Grid environment - 8-connected
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (node[0] + dx, node[1] + dy)
                    if graph.is_valid(neighbor):
                        neighbors.append(neighbor)
            return neighbors
    
    def _reconstruct_path(self, current: Tuple) -> List[Tuple]:
        """Reconstruct path from parent pointers"""
        path = []
        while current is not None:
            path.append(current)
            current = self.parent.get(current)
        return list(reversed(path))
    
    def _best_partial_path(self, goal: Tuple) -> List[Tuple]:
        """Return best partial path if timeout"""
        if not self.g_score:
            return []
        
        # Find node closest to goal in explored set
        best_node = min(self.closed_set, 
                       key=lambda n: np.sqrt((goal[0]-n[0])**2 + (goal[1]-n[1])**2))
        return self._reconstruct_path(best_node)
    
    def _path_length(self, path: List[Tuple]) -> float:
        """Calculate total Euclidean path length"""
        if len(path) < 2:
            return 0.0
        
        length = 0.0
        for i in range(len(path) - 1):
            length += np.sqrt(
                (path[i+1][0] - path[i][0])**2 + 
                (path[i+1][1] - path[i][1])**2
            )
        return length
