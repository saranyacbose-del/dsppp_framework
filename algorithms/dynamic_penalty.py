"""
Dynamic Penalty Cost System with Kalman Filtering
Real-time obstacle trajectory prediction and penalty map updates
"""

import numpy as np
from typing import List, Dict, Tuple
from filterpy.kalman import KalmanFilter
import time


class DynamicPenaltySystem:
    """
    Asynchronous dynamic penalty update system
    Maintains obstacle predictions and penalty costs at 10 Hz
    """
    
    def __init__(self, config: Dict):
        """
        Initialize penalty system
        
        Args:
            config: Configuration with penalty parameters
        """
        self.lambda_base = config.get('lambda_base', 5.0)
        self.sigma = config.get('sigma', 2.0)
        self.delta_t = config.get('delta_t', 1.0)
        self.t_pred = config.get('t_pred', 2.0)
        self.decay_rate = config.get('decay_rate', 0.05)
        self.v_max = config.get('v_max', 10.0)
        self.update_freq = config.get('update_freq', 10.0)
        
        self.obstacle_filters = {}
        self.penalty_map = {}
        self.last_update = time.time()
        
    def initialize_obstacle(self, obs_id: int, state: Dict) -> None:
        """
        Initialize Kalman filter for new obstacle
        
        Args:
            obs_id: Obstacle identifier
            state: Initial state dict with px, py, vx, vy
        """
        # Create Kalman filter for constant velocity model
        kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (Equation 3)
        dt = 1.0 / self.update_freq
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Observation matrix
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance (Equation 5)
        kf.Q = np.diag([0.1, 0.1, 0.5, 0.5])
        
        # Measurement noise covariance
        kf.R = np.diag([0.5, 0.5])
        
        # Initial state
        kf.x = np.array([
            state['px'],
            state['py'],
            state.get('vx', 0.0),
            state.get('vy', 0.0)
        ])
        
        # Initial covariance
        kf.P = np.eye(4) * 1.0
        
        self.obstacle_filters[obs_id] = kf
    
    def update_obstacle(self, obs_id: int, measurement: Tuple[float, float]) -> None:
        """
        Update obstacle state with new measurement
        
        Args:
            obs_id: Obstacle identifier
            measurement: (px, py) position measurement
        """
        if obs_id not in self.obstacle_filters:
            # Initialize if first time seeing this obstacle
            self.initialize_obstacle(obs_id, {
                'px': measurement[0],
                'py': measurement[1]
            })
            return
        
        kf = self.obstacle_filters[obs_id]
        
        # Prediction step (Equations 4-5)
        kf.predict()
        
        # Update step (Equations 6-8)
        kf.update(np.array(measurement))
    
    def predict_trajectory(self, obs_id: int, horizon: float = None) -> List[Tuple]:
        """
        Predict obstacle trajectory over time horizon
        
        Args:
            obs_id: Obstacle identifier
            horizon: Prediction time horizon (default: self.t_pred)
            
        Returns:
            List of (px, py, t) predicted positions
        """
        if horizon is None:
            horizon = self.t_pred
        
        if obs_id not in self.obstacle_filters:
            return []
        
        kf = self.obstacle_filters[obs_id]
        
        # Current state
        x = kf.x.copy()
        F = kf.F
        
        trajectory = []
        dt = 1.0 / self.update_freq
        steps = int(horizon / dt)
        
        for i in range(steps):
            t = i * dt
            # Propagate state forward
            x = F @ x
            trajectory.append((x[0], x[1], t))
        
        return trajectory
    
    def calculate_penalty_map(self, obstacles: List[Dict], 
                             active_region: List[Tuple]) -> Dict:
        """
        Calculate penalty costs for all nodes in active region
        
        Args:
            obstacles: List of obstacle dicts with id and measurements
            active_region: List of (x, y) coordinates to update
            
        Returns:
            penalty_map: Dict mapping (x,y) -> penalty_cost
        """
        penalty_map = {}
        
        # Update obstacle filters
        for obs in obstacles:
            obs_id = obs['id']
            measurement = (obs['px'], obs['py'])
            self.update_obstacle(obs_id, measurement)
        
        # Calculate penalties for active region
        for node in active_region:
            penalty = 0.0
            
            for obs in obstacles:
                obs_id = obs['id']
                
                # Get predicted trajectory
                trajectory = self.predict_trajectory(obs_id, self.t_pred)
                
                if not trajectory:
                    continue
                
                # Find minimum distance to trajectory
                min_dist = float('inf')
                for pred_pos in trajectory:
                    dist = np.sqrt(
                        (node[0] - pred_pos[0])**2 + 
                        (node[1] - pred_pos[1])**2
                    )
                    min_dist = min(min_dist, dist)
                
                # Calculate obstacle weight (Equation 18)
                kf = self.obstacle_filters[obs_id]
                v_mag = np.sqrt(kf.x[2]**2 + kf.x[3]**2)
                lambda_o = self.lambda_base * (1 + v_mag / self.v_max)
                
                # Add penalty contribution (Equation 17)
                penalty += lambda_o * np.exp(-min_dist**2 / (2 * self.sigma**2))
            
            penalty_map[node] = penalty
        
        return penalty_map
    
    def update(self, obstacles: List[Dict], 
              active_region: List[Tuple]) -> Dict:
        """
        Asynchronous penalty map update (Algorithm 2)
        
        Args:
            obstacles: Current obstacle detections
            active_region: Nodes to update penalties for
            
        Returns:
            Updated penalty map
        """
        current_time = time.time()
        dt = current_time - self.last_update
        
        # Apply temporal decay to existing penalties
        for node in self.penalty_map:
            self.penalty_map[node] *= np.exp(-self.decay_rate * dt)
        
        # Calculate new penalties
        new_penalties = self.calculate_penalty_map(obstacles, active_region)
        
        # Merge with existing (take maximum)
        for node, penalty in new_penalties.items():
            self.penalty_map[node] = max(
                self.penalty_map.get(node, 0.0),
                penalty
            )
        
        self.last_update = current_time
        
        return self.penalty_map
    
    def get_penalty(self, node: Tuple) -> float:
        """Get penalty cost for specific node"""
        return self.penalty_map.get(node, 0.0)
    
    def get_obstacle_state(self, obs_id: int) -> Optional[np.ndarray]:
        """Get current state estimate for obstacle"""
        if obs_id in self.obstacle_filters:
            return self.obstacle_filters[obs_id].x.copy()
        return None
