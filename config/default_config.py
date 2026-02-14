"""
Default Configuration for DSPPP Framework
"""

# Algorithm Parameters
ALGORITHM_CONFIG = {
    # Dynamic Penalty System
    'lambda_base': 5.0,        # Base obstacle penalty weight
    'sigma': 2.0,              # Penalty spatial spread (grid units)
    'delta_t': 1.0,            # Planning lookahead time (seconds)
    't_pred': 2.0,             # Kalman prediction horizon (seconds)
    'v_max': 10.0,             # Maximum expected obstacle velocity (m/s)
    'decay_rate': 0.05,        # Temporal penalty decay rate (1/s)
    'update_freq': 10.0,       # Penalty map update frequency (Hz)
    
    # Improved A* Parameters
    'gamma': 0.15,             # Heuristic density sensitivity
    'beta': 0.1,               # Path smoothness weight
    'w_base': 1.0,             # Base semantic weight
    't_max': 5.0,              # Maximum planning time budget (seconds)
    'd_safety': 2.0,           # Safety margin (grid units)
    'r_sense': 5.0,            # Obstacle density sensing radius
    
    # PPE Metric Weights
    'w_length': 0.4,           # Length importance weight
    'w_semantic': 0.3,         # Semantic importance weight
    'w_safety': 0.3,           # Safety importance weight
}

# Environment Parameters
ENVIRONMENT_CONFIG = {
    'grid_width': 100,
    'grid_height': 100,
    'obstacle_density': 0.2,
    'n_dynamic_obstacles': 10,
}

# Experiment Parameters
EXPERIMENT_CONFIG = {
    'n_trials': 30,
    'random_seed': 42,
    'save_visualizations': True,
    'save_results': True,
}
