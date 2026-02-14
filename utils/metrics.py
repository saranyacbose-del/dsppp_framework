"""
Performance Metrics for Path Planning Evaluation
Implements PPE and other metrics from the paper
"""

import numpy as np
from typing import List, Tuple, Dict


class PathPlanningMetrics:
    """Calculate comprehensive path planning metrics"""
    
    def __init__(self, config: Dict = None):
        """Initialize with metric weights"""
        if config is None:
            config = {}
        
        self.w_length = config.get('w_length', 0.4)
        self.w_semantic = config.get('w_semantic', 0.3)
        self.w_safety = config.get('w_safety', 0.3)
    
    def calculate_all(self, path: List[Tuple], start: Tuple, goal: Tuple,
                     obstacles: List[Dict] = None,
                     semantic_scores: Dict = None,
                     computation_time: float = 0.0) -> Dict:
        """
        Calculate all metrics for a path
        
        Returns dictionary with all metric values
        """
        metrics = {}
        
        # Path length
        metrics['path_length'] = self.path_length(path)
        metrics['optimal_length'] = self.euclidean_distance(start, goal)
        metrics['length_ratio'] = self.length_ratio(path, start, goal)
        
        # Path smoothness
        metrics['smoothness'] = self.path_smoothness(path)
        
        # Safety margin
        if obstacles:
            metrics['safety_margin'] = self.safety_margin(path, obstacles)
        else:
            metrics['safety_margin'] = float('inf')
        
        # Semantic alignment
        if semantic_scores:
            metrics['semantic_score'] = self.semantic_alignment(path, semantic_scores)
        else:
            metrics['semantic_score'] = 0.0
        
        # Computation time
        metrics['computation_time'] = computation_time
        
        # PPE (Personalized Path Efficiency)
        metrics['ppe'] = self.personalized_path_efficiency(
            metrics['length_ratio'],
            metrics['semantic_score'],
            metrics['safety_margin'],
            computation_time
        )
        
        return metrics
    
    def path_length(self, path: List[Tuple]) -> float:
        """Calculate total Euclidean path length"""
        if len(path) < 2:
            return 0.0
        
        length = 0.0
        for i in range(len(path) - 1):
            length += self.euclidean_distance(path[i], path[i+1])
        
        return length
    
    def euclidean_distance(self, p1: Tuple, p2: Tuple) -> float:
        """Euclidean distance between two points"""
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def length_ratio(self, path: List[Tuple], start: Tuple, goal: Tuple) -> float:
        """
        Length ratio metric (Equation 30)
        R_length = L_optimal / L_actual
        """
        l_optimal = self.euclidean_distance(start, goal)
        l_actual = self.path_length(path)
        
        if l_actual == 0:
            return 0.0
        
        return l_optimal / l_actual
    
    def path_smoothness(self, path: List[Tuple]) -> float:
        """
        Path smoothness measured by average angular deviation
        Lower is better (smoother)
        """
        if len(path) < 3:
            return 0.0
        
        angles = []
        for i in range(1, len(path) - 1):
            # Vectors
            v1 = np.array([path[i][0] - path[i-1][0], 
                          path[i][1] - path[i-1][1]])
            v2 = np.array([path[i+1][0] - path[i][0], 
                          path[i+1][1] - path[i][1]])
            
            # Angle between vectors
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angles.append(np.degrees(angle))
        
        if not angles:
            return 0.0
        
        return np.mean(angles)
    
    def safety_margin(self, path: List[Tuple], obstacles: List[Dict],
                     threshold: float = 2.0) -> float:
        """
        Safety margin metric (Equation 32)
        R_safety = min(d(p,o)) / d_threshold
        """
        if not obstacles or not path:
            return float('inf')
        
        min_distances = []
        
        for point in path:
            min_dist = float('inf')
            for obs in obstacles:
                dist = self.euclidean_distance(
                    point, 
                    (obs['px'], obs['py'])
                )
                min_dist = min(min_dist, dist)
            min_distances.append(min_dist)
        
        overall_min = min(min_distances) if min_distances else float('inf')
        
        return overall_min / threshold
    
    def semantic_alignment(self, path: List[Tuple], 
                          semantic_scores: Dict) -> float:
        """
        Semantic alignment score (Equation 31)
        Measures how well path aligns with user preferences
        """
        if len(path) < 2:
            return 0.0
        
        total_score = 0.0
        
        for i in range(len(path) - 1):
            edge = (path[i], path[i+1])
            score = semantic_scores.get(edge, 0.0)
            total_score += score
        
        # Normalize by path length
        return total_score / len(path)
    
    def personalized_path_efficiency(self, length_ratio: float,
                                    semantic_score: float,
                                    safety_ratio: float,
                                    computation_time: float) -> float:
        """
        PPE metric (Equation 29)
        
        PPE = (W_length * R_length + W_semantic * R_semantic + W_safety * R_safety) / T_computation
        """
        if computation_time == 0:
            computation_time = 0.001  # Avoid division by zero
        
        # Convert to milliseconds
        time_ms = computation_time * 1000.0
        
        numerator = (
            self.w_length * length_ratio + 
            self.w_semantic * semantic_score + 
            self.w_safety * min(safety_ratio, 2.0)  # Cap safety ratio
        )
        
        ppe = numerator / time_ms
        
        return ppe


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size
    
    Args:
        group1: First group of measurements
        group2: Second group of measurements
        
    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return d
