"""
Baseline Path Planning Algorithms
Standard A*, Dijkstra, RRT*, ACO for performance comparison
"""

import numpy as np
import heapq
import networkx as nx
from typing import List, Tuple, Dict
import time


class StandardAStar:
    """Standard A* algorithm without enhancements"""
    
    def __init__(self):
        self.open_set = []
        self.closed_set = set()
        self.g_score = {}
        self.f_score = {}
        self.parent = {}
    
    def search(self, start: Tuple, goal: Tuple, graph) -> Tuple[List, Dict]:
        """Execute standard A* search"""
        t_start = time.perf_counter()
        
        self.open_set = []
        self.closed_set = set()
        self.g_score = {start: 0}
        self.f_score = {start: self._heuristic(start, goal)}
        self.parent = {start: None}
        
        heapq.heappush(self.open_set, (self.f_score[start], start))
        
        nodes_explored = 0
        
        while self.open_set:
            _, current = heapq.heappop(self.open_set)
            
            if current in self.closed_set:
                continue
            
            nodes_explored += 1
            
            if current == goal:
                path = self._reconstruct_path(current)
                return path, {
                    'success': True,
                    'time': time.perf_counter() - t_start,
                    'nodes_explored': nodes_explored
                }
            
            self.closed_set.add(current)
            
            for neighbor in self._get_neighbors(current, graph):
                if neighbor in self.closed_set:
                    continue
                
                tentative_g = self.g_score[current] + self._distance(current, neighbor)
                
                if neighbor not in self.g_score or tentative_g < self.g_score[neighbor]:
                    self.parent[neighbor] = current
                    self.g_score[neighbor] = tentative_g
                    self.f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(self.open_set, (self.f_score[neighbor], neighbor))
        
        return [], {'success': False, 'time': time.perf_counter() - t_start, 
                   'nodes_explored': nodes_explored}
    
    def _heuristic(self, n: Tuple, goal: Tuple) -> float:
        """Euclidean distance heuristic"""
        return np.sqrt((goal[0] - n[0])**2 + (goal[1] - n[1])**2)
    
    def _distance(self, n1: Tuple, n2: Tuple) -> float:
        """Euclidean distance"""
        return np.sqrt((n2[0] - n1[0])**2 + (n2[1] - n1[1])**2)
    
    def _get_neighbors(self, node: Tuple, graph) -> List[Tuple]:
        """Get 8-connected neighbors for grid"""
        if hasattr(graph, 'neighbors'):
            return list(graph.neighbors(node))
        else:
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


class DijkstraPlanner:
    """Dijkstra's algorithm"""
    
    def search(self, start: Tuple, goal: Tuple, graph) -> Tuple[List, Dict]:
        """Execute Dijkstra search"""
        t_start = time.perf_counter()
        
        # Convert to NetworkX graph if needed
        if not hasattr(graph, 'neighbors'):
            G = self._grid_to_graph(graph)
        else:
            G = graph
        
        try:
            path = nx.shortest_path(G, source=start, target=goal, weight='weight')
            nodes_explored = len(G.nodes())
            
            return path, {
                'success': True,
                'time': time.perf_counter() - t_start,
                'nodes_explored': nodes_explored
            }
        except nx.NetworkXNoPath:
            return [], {
                'success': False,
                'time': time.perf_counter() - t_start,
                'nodes_explored': len(G.nodes())
            }
    
    def _grid_to_graph(self, grid):
        """Convert grid to NetworkX graph"""
        G = nx.Graph()
        for i in range(grid.width):
            for j in range(grid.height):
                if grid.is_valid((i, j)):
                    G.add_node((i, j))
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            neighbor = (i + dx, j + dy)
                            if grid.is_valid(neighbor):
                                dist = np.sqrt(dx**2 + dy**2)
                                G.add_edge((i, j), neighbor, weight=dist)
        return G


class RRTStarPlanner:
    """RRT* (Rapidly-exploring Random Tree Star) algorithm"""
    
    def __init__(self, max_iterations=10000, goal_bias=0.05):
        self.max_iterations = max_iterations
        self.goal_bias = goal_bias
        self.gamma = 2.5
    
    def search(self, start: Tuple, goal: Tuple, graph) -> Tuple[List, Dict]:
        """Execute RRT* search"""
        t_start = time.perf_counter()
        
        # Tree structure
        nodes = [start]
        parent = {start: None}
        cost = {start: 0.0}
        
        goal_reached = False
        goal_node = None
        
        for iteration in range(self.max_iterations):
            # Sample random point
            if np.random.random() < self.goal_bias:
                rand_point = goal
            else:
                rand_point = self._sample_free(graph)
            
            # Find nearest node
            nearest = min(nodes, key=lambda n: self._distance(n, rand_point))
            
            # Steer towards random point
            new_node = self._steer(nearest, rand_point, step_size=1.0)
            
            # Check collision
            if not graph.is_valid(new_node):
                continue
            
            # Find nearby nodes for rewiring
            radius = min(self.gamma * np.sqrt(np.log(len(nodes)) / len(nodes)), 2.0)
            nearby = [n for n in nodes if self._distance(n, new_node) <= radius]
            
            # Choose best parent
            best_parent = nearest
            best_cost = cost[nearest] + self._distance(nearest, new_node)
            
            for near in nearby:
                potential_cost = cost[near] + self._distance(near, new_node)
                if potential_cost < best_cost:
                    best_parent = near
                    best_cost = potential_cost
            
            # Add new node
            nodes.append(new_node)
            parent[new_node] = best_parent
            cost[new_node] = best_cost
            
            # Rewire tree
            for near in nearby:
                potential_cost = cost[new_node] + self._distance(new_node, near)
                if potential_cost < cost[near]:
                    parent[near] = new_node
                    cost[near] = potential_cost
            
            # Check if goal reached
            if self._distance(new_node, goal) < 1.0:
                goal_reached = True
                goal_node = new_node
        
        # Reconstruct path
        if goal_reached:
            path = []
            current = goal_node
            while current is not None:
                path.append(current)
                current = parent.get(current)
            path = list(reversed(path))
            
            return path, {
                'success': True,
                'time': time.perf_counter() - t_start,
                'nodes_explored': len(nodes)
            }
        
        return [], {
            'success': False,
            'time': time.perf_counter() - t_start,
            'nodes_explored': len(nodes)
        }
    
    def _sample_free(self, graph) -> Tuple:
        """Sample random free configuration"""
        while True:
            x = np.random.randint(0, graph.width)
            y = np.random.randint(0, graph.height)
            if graph.is_valid((x, y)):
                return (x, y)
    
    def _steer(self, from_node: Tuple, to_node: Tuple, step_size: float) -> Tuple:
        """Steer from one node towards another"""
        dx = to_node[0] - from_node[0]
        dy = to_node[1] - from_node[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist <= step_size:
            return to_node
        
        ratio = step_size / dist
        new_x = from_node[0] + dx * ratio
        new_y = from_node[1] + dy * ratio
        
        return (int(round(new_x)), int(round(new_y)))
    
    def _distance(self, n1: Tuple, n2: Tuple) -> float:
        """Euclidean distance"""
        return np.sqrt((n2[0] - n1[0])**2 + (n2[1] - n1[1])**2)


class ACOPlanner:
    """Ant Colony Optimization algorithm"""
    
    def __init__(self, n_ants=50, n_iterations=100, alpha=1.0, beta=2.0, rho=0.1):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Heuristic importance
        self.rho = rho      # Evaporation rate
    
    def search(self, start: Tuple, goal: Tuple, graph) -> Tuple[List, Dict]:
        """Execute ACO search"""
        t_start = time.perf_counter()
        
        # Initialize pheromones
        pheromone = {}
        
        best_path = None
        best_length = float('inf')
        
        for iteration in range(self.n_iterations):
            paths = []
            
            for ant in range(self.n_ants):
                path = self._construct_path(start, goal, graph, pheromone)
                if path:
                    paths.append(path)
                    length = self._path_length(path)
                    if length < best_length:
                        best_length = length
                        best_path = path
            
            # Update pheromones
            self._update_pheromones(pheromone, paths)
        
        if best_path:
            return best_path, {
                'success': True,
                'time': time.perf_counter() - t_start,
                'nodes_explored': self.n_ants * self.n_iterations
            }
        
        return [], {
            'success': False,
            'time': time.perf_counter() - t_start,
            'nodes_explored': self.n_ants * self.n_iterations
        }
    
    def _construct_path(self, start: Tuple, goal: Tuple, graph, pheromone) -> List:
        """Construct path using probabilistic transition"""
        path = [start]
        current = start
        visited = {start}
        max_steps = 1000
        
        for _ in range(max_steps):
            if current == goal:
                return path
            
            neighbors = [n for n in self._get_neighbors(current, graph) 
                        if n not in visited]
            
            if not neighbors:
                return None
            
            # Calculate transition probabilities
            probs = []
            for neighbor in neighbors:
                edge = (current, neighbor)
                tau = pheromone.get(edge, 1.0)  # Pheromone
                eta = 1.0 / (self._distance(current, neighbor) + 0.01)  # Heuristic
                probs.append(tau**self.alpha * eta**self.beta)
            
            # Normalize
            total = sum(probs)
            probs = [p / total for p in probs]
            
            # Select next node
            next_node = np.random.choice(neighbors, p=probs)
            path.append(next_node)
            visited.add(next_node)
            current = next_node
        
        return None
    
    def _update_pheromones(self, pheromone: Dict, paths: List) -> None:
        """Update pheromone trails"""
        # Evaporation
        for edge in list(pheromone.keys()):
            pheromone[edge] *= (1 - self.rho)
        
        # Deposition
        for path in paths:
            length = self._path_length(path)
            deposit = 1.0 / (length + 0.01)
            
            for i in range(len(path) - 1):
                edge = (path[i], path[i+1])
                pheromone[edge] = pheromone.get(edge, 0.0) + deposit
    
    def _get_neighbors(self, node: Tuple, graph) -> List[Tuple]:
        """Get neighboring nodes"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (node[0] + dx, node[1] + dy)
                if graph.is_valid(neighbor):
                    neighbors.append(neighbor)
        return neighbors
    
    def _distance(self, n1: Tuple, n2: Tuple) -> float:
        """Euclidean distance"""
        return np.sqrt((n2[0] - n1[0])**2 + (n2[1] - n1[1])**2)
    
    def _path_length(self, path: List) -> float:
        """Calculate total path length"""
        if len(path) < 2:
            return 0.0
        length = 0.0
        for i in range(len(path) - 1):
            length += self._distance(path[i], path[i+1])
        return length
