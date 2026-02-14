"""
Complete DSPPP Framework Demo
Demonstrates all capabilities on grid environment
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from environment.grid_env import GridEnvironment
from algorithms.improved_astar import ImprovedAStar
from algorithms.baseline_planners import StandardAStar, DijkstraPlanner
from algorithms.dynamic_penalty import DynamicPenaltySystem
from utils.metrics import PathPlanningMetrics
import time


def run_demo():
    """Run complete DSPPP demonstration"""
    
    print("=" * 70)
    print("DSPPP Framework Demonstration")
    print("Dynamic Semantic Personalized Path Planning")
    print("=" * 70)
    
    # Configuration
    config = {
        'lambda_base': 5.0,
        'sigma': 2.0,
        'delta_t': 1.0,
        'gamma': 0.15,
        'beta': 0.1,
        't_max': 5.0,
        't_pred': 2.0,
        'update_freq': 10.0,
        'v_max': 10.0,
        'decay_rate': 0.05
    }
    
    # Create environment
    print("\n1. Creating 50x50 grid environment with 20% obstacles...")
    env = GridEnvironment(width=50, height=50, obstacle_density=0.20)
    
    # Add dynamic obstacles
    print("2. Adding 10 dynamic obstacles...")
    for i in range(10):
        px = np.random.uniform(10, 40)
        py = np.random.uniform(10, 40)
        vx = np.random.uniform(-2.5, 2.5)
        vy = np.random.uniform(-2.5, 2.5)
        env.add_dynamic_obstacle(i, px, py, vx, vy)
    
    # Set start and goal
    start = (5, 5)
    goal = (45, 45)
    print(f"3. Start: {start}, Goal: {goal}")
    
    # Initialize planners
    print("\n4. Initializing planners...")
    dsppp = ImprovedAStar(config)
    standard_astar = StandardAStar()
    dijkstra = DijkstraPlanner()
    
    # Initialize penalty system
    penalty_system = DynamicPenaltySystem(config)
    
    # Get active region for penalty calculation
    initial_path = [start, goal]
    active_region = env.get_active_region(initial_path, expansion=5.0)
    
    # Calculate penalty map
    print("5. Computing dynamic penalty map with Kalman filtering...")
    penalty_map = penalty_system.update(env.dynamic_obstacles, active_region)
    
    # Initialize metrics
    metrics_calc = PathPlanningMetrics(config)
    
    # Run algorithms
    print("\n6. Running path planning algorithms...")
    print("-" * 70)
    
    results = {}
    
    # DSPPP
    print("   Running DSPPP (Improved A*)...")
    t_start = time.perf_counter()
    path_dsppp, stats_dsppp = dsppp.search(
        start, goal, env,
        obstacles=env.dynamic_obstacles,
        penalty_map=penalty_map,
        semantic_costs=None
    )
    t_dsppp = time.perf_counter() - t_start
    
    if stats_dsppp['success']:
        metrics_dsppp = metrics_calc.calculate_all(
            path_dsppp, start, goal,
            obstacles=env.dynamic_obstacles,
            computation_time=t_dsppp
        )
        results['DSPPP'] = {
            'path': path_dsppp,
            'stats': stats_dsppp,
            'metrics': metrics_dsppp
        }
        print(f"   ✓ Success! Time: {t_dsppp*1000:.1f}ms, "
              f"Length: {metrics_dsppp['path_length']:.1f}, "
              f"PPE: {metrics_dsppp['ppe']:.4f}")
    else:
        print(f"   ✗ Failed: {stats_dsppp.get('reason', 'unknown')}")
    
    # Standard A*
    print("   Running Standard A*...")
    t_start = time.perf_counter()
    path_astar, stats_astar = standard_astar.search(start, goal, env)
    t_astar = time.perf_counter() - t_start
    
    if stats_astar['success']:
        metrics_astar = metrics_calc.calculate_all(
            path_astar, start, goal,
            obstacles=env.dynamic_obstacles,
            computation_time=t_astar
        )
        results['Standard A*'] = {
            'path': path_astar,
            'stats': stats_astar,
            'metrics': metrics_astar
        }
        print(f"   ✓ Success! Time: {t_astar*1000:.1f}ms, "
              f"Length: {metrics_astar['path_length']:.1f}, "
              f"PPE: {metrics_astar['ppe']:.4f}")
    else:
        print(f"   ✗ Failed")
    
    # Dijkstra
    print("   Running Dijkstra...")
    t_start = time.perf_counter()
    path_dijkstra, stats_dijkstra = dijkstra.search(start, goal, env)
    t_dijkstra = time.perf_counter() - t_start
    
    if stats_dijkstra['success']:
        metrics_dijkstra = metrics_calc.calculate_all(
            path_dijkstra, start, goal,
            obstacles=env.dynamic_obstacles,
            computation_time=t_dijkstra
        )
        results['Dijkstra'] = {
            'path': path_dijkstra,
            'stats': stats_dijkstra,
            'metrics': metrics_dijkstra
        }
        print(f"   ✓ Success! Time: {t_dijkstra*1000:.1f}ms, "
              f"Length: {metrics_dijkstra['path_length']:.1f}, "
              f"PPE: {metrics_dijkstra['ppe']:.4f}")
    else:
        print(f"   ✗ Failed")
    
    # Print comparison
    print("\n7. Performance Comparison:")
    print("-" * 70)
    print(f"{'Metric':<25} {'DSPPP':<15} {'Standard A*':<15} {'Dijkstra':<15}")
    print("-" * 70)
    
    if 'DSPPP' in results and 'Standard A*' in results:
        metrics_names = [
            ('Computation Time (ms)', 'computation_time', 1000),
            ('Path Length', 'path_length', 1),
            ('Path Smoothness (°)', 'smoothness', 1),
            ('Safety Margin', 'safety_margin', 1),
            ('PPE Metric', 'ppe', 1)
        ]
        
        for name, key, scale in metrics_names:
            dsppp_val = results['DSPPP']['metrics'][key] * scale
            astar_val = results['Standard A*']['metrics'][key] * scale
            dijkstra_val = results.get('Dijkstra', {}).get('metrics', {}).get(key, 0) * scale
            
            print(f"{name:<25} {dsppp_val:<15.2f} {astar_val:<15.2f} {dijkstra_val:<15.2f}")
        
        # Calculate improvements
        print("\n8. DSPPP Improvements over Standard A*:")
        print("-" * 70)
        
        time_improvement = ((t_astar - t_dsppp) / t_astar) * 100
        smooth_improvement = ((metrics_astar['smoothness'] - metrics_dsppp['smoothness']) / 
                             metrics_astar['smoothness']) * 100
        ppe_improvement = ((metrics_dsppp['ppe'] - metrics_astar['ppe']) / 
                          metrics_astar['ppe']) * 100
        
        print(f"   Computation Time: {time_improvement:+.1f}% faster")
        print(f"   Path Smoothness:  {smooth_improvement:+.1f}% smoother")
        print(f"   PPE Metric:       {ppe_improvement:+.1f}% higher")
    
    # Visualize
    print("\n9. Generating visualizations...")
    
    if 'DSPPP' in results:
        env.visualize(
            path=results['DSPPP']['path'],
            start=start,
            goal=goal,
            penalty_map=penalty_map,
            save_path='output_dsppp_demo.png'
        )
        print("   ✓ Saved: output_dsppp_demo.png")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    np.random.seed(42)  # For reproducibility
    run_demo()
