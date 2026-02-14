"""
Comprehensive Benchmark Script
Evaluates DSPPP vs baselines across multiple scenarios
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

from environment.grid_env import GridEnvironment
from algorithms.improved_astar import ImprovedAStar
from algorithms.baseline_planners import StandardAStar, DijkstraPlanner, RRTStarPlanner, ACOPlanner
from algorithms.dynamic_penalty import DynamicPenaltySystem
from utils.metrics import PathPlanningMetrics, cohens_d


def run_single_trial(env, start, goal, planner_name, planner, 
                    penalty_system=None, config=None):
    """Run single planning trial"""
    
    # Get penalty map if using DSPPP
    penalty_map = None
    if planner_name == 'DSPPP' and penalty_system:
        active_region = env.get_active_region([start, goal], expansion=5.0)
        penalty_map = penalty_system.update(env.dynamic_obstacles, active_region)
    
    # Run planner
    t_start = time.perf_counter()
    
    if planner_name == 'DSPPP':
        path, stats = planner.search(
            start, goal, env,
            obstacles=env.dynamic_obstacles,
            penalty_map=penalty_map,
            semantic_costs=None
        )
    else:
        path, stats = planner.search(start, goal, env)
    
    t_elapsed = time.perf_counter() - t_start
    
    # Calculate metrics
    metrics_calc = PathPlanningMetrics(config or {})
    metrics = metrics_calc.calculate_all(
        path, start, goal,
        obstacles=env.dynamic_obstacles,
        computation_time=t_elapsed
    )
    
    return {
        'success': stats.get('success', False),
        'time': t_elapsed * 1000,  # Convert to ms
        'path_length': metrics['path_length'],
        'smoothness': metrics['smoothness'],
        'safety_margin': metrics['safety_margin'],
        'ppe': metrics['ppe'],
        'nodes_explored': stats.get('nodes_explored', 0)
    }


def run_benchmark(n_trials=30, grid_sizes=[50], obstacle_densities=[0.2]):
    """
    Run comprehensive benchmark
    
    Args:
        n_trials: Number of trials per configuration
        grid_sizes: List of grid sizes to test
        obstacle_densities: List of obstacle densities to test
    """
    
    print("=" * 80)
    print("DSPPP Framework Comprehensive Benchmark")
    print("=" * 80)
    
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
    
    # Initialize planners
    planners = {
        'DSPPP': ImprovedAStar(config),
        'Standard A*': StandardAStar(),
        'Dijkstra': DijkstraPlanner(),
        'RRT*': RRTStarPlanner(max_iterations=5000),
        'ACO': ACOPlanner(n_ants=30, n_iterations=50)
    }
    
    # Results storage
    all_results = []
    
    # Run experiments
    total_experiments = len(grid_sizes) * len(obstacle_densities) * n_trials * len(planners)
    pbar = tqdm(total=total_experiments, desc="Running benchmark")
    
    for grid_size in grid_sizes:
        for obs_density in obstacle_densities:
            print(f"\n\nTesting: Grid {grid_size}x{grid_size}, {int(obs_density*100)}% obstacles")
            print("-" * 80)
            
            for trial in range(n_trials):
                # Create environment
                np.random.seed(trial)  # Different seed per trial
                env = GridEnvironment(grid_size, grid_size, obs_density)
                
                # Add dynamic obstacles
                n_obstacles = 10
                for i in range(n_obstacles):
                    px = np.random.uniform(grid_size * 0.2, grid_size * 0.8)
                    py = np.random.uniform(grid_size * 0.2, grid_size * 0.8)
                    vx = np.random.normal(0, 1.5)
                    vy = np.random.normal(0, 1.5)
                    env.add_dynamic_obstacle(i, px, py, vx, vy)
                
                # Random start and goal
                start = env.get_random_free_position()
                goal = env.get_random_free_position()
                
                # Ensure sufficient separation
                while np.linalg.norm(np.array(goal) - np.array(start)) < grid_size * 0.7:
                    goal = env.get_random_free_position()
                
                # Initialize penalty system for DSPPP
                penalty_system = DynamicPenaltySystem(config)
                
                # Test each planner
                for planner_name, planner in planners.items():
                    try:
                        result = run_single_trial(
                            env, start, goal, planner_name, planner,
                            penalty_system if planner_name == 'DSPPP' else None,
                            config
                        )
                        
                        result.update({
                            'planner': planner_name,
                            'grid_size': grid_size,
                            'obs_density': obs_density,
                            'trial': trial
                        })
                        
                        all_results.append(result)
                        
                    except Exception as e:
                        print(f"Error in {planner_name}: {str(e)}")
                        result = {
                            'planner': planner_name,
                            'grid_size': grid_size,
                            'obs_density': obs_density,
                            'trial': trial,
                            'success': False,
                            'time': 0,
                            'path_length': 0,
                            'smoothness': 0,
                            'safety_margin': 0,
                            'ppe': 0,
                            'nodes_explored': 0
                        }
                        all_results.append(result)
                    
                    pbar.update(1)
    
    pbar.close()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Save results
    df.to_csv('benchmark_results.csv', index=False)
    print("\n✓ Results saved to: benchmark_results.csv")
    
    # Generate analysis
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    for planner_name in planners.keys():
        planner_df = df[df['planner'] == planner_name]
        successful = planner_df[planner_df['success'] == True]
        
        print(f"\n{planner_name}:")
        print(f"  Success Rate: {len(successful)/len(planner_df)*100:.1f}%")
        
        if len(successful) > 0:
            print(f"  Avg Time: {successful['time'].mean():.1f} ± {successful['time'].std():.1f} ms")
            print(f"  Avg Path Length: {successful['path_length'].mean():.1f} ± {successful['path_length'].std():.1f}")
            print(f"  Avg Smoothness: {successful['smoothness'].mean():.1f}° ± {successful['smoothness'].std():.1f}°")
            print(f"  Avg PPE: {successful['ppe'].mean():.4f} ± {successful['ppe'].std():.4f}")
    
    # Statistical analysis
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS (DSPPP vs Standard A*)")
    print("=" * 80)
    
    dsppp_data = df[df['planner'] == 'DSPPP']
    astar_data = df[df['planner'] == 'Standard A*']
    
    # Filter successful trials
    dsppp_success = dsppp_data[dsppp_data['success'] == True]
    astar_success = astar_data[astar_data['success'] == True]
    
    if len(dsppp_success) > 0 and len(astar_success) > 0:
        metrics_to_compare = ['time', 'smoothness', 'ppe', 'nodes_explored']
        
        for metric in metrics_to_compare:
            d = cohens_d(
                dsppp_success[metric].values,
                astar_success[metric].values
            )
            
            interpretation = ""
            if abs(d) < 0.5:
                interpretation = "small"
            elif abs(d) < 1.5:
                interpretation = "medium"  
            elif abs(d) < 3.0:
                interpretation = "large"
            else:
                interpretation = "very large"
            
            print(f"\n{metric.upper()}:")
            print(f"  DSPPP:  {dsppp_success[metric].mean():.2f} ± {dsppp_success[metric].std():.2f}")
            print(f"  A*:     {astar_success[metric].mean():.2f} ± {astar_success[metric].std():.2f}")
            print(f"  Cohen's d: {d:.2f} ({interpretation} effect)")
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    generate_comparison_plots(df)
    
    print("\n✓ Benchmark completed!")
    print("=" * 80)
    
    return df


def generate_comparison_plots(df):
    """Generate comparison plots"""
    
    # Filter successful trials
    df_success = df[df['success'] == True]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Computation Time Comparison
    ax = axes[0, 0]
    data_time = df_success.groupby('planner')['time'].apply(list)
    ax.boxplot([data_time[p] for p in data_time.index], labels=data_time.index)
    ax.set_ylabel('Computation Time (ms)')
    ax.set_title('Computation Time Comparison')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Path Smoothness
    ax = axes[0, 1]
    data_smooth = df_success.groupby('planner')['smoothness'].apply(list)
    ax.boxplot([data_smooth[p] for p in data_smooth.index], labels=data_smooth.index)
    ax.set_ylabel('Path Smoothness (degrees)')
    ax.set_title('Path Smoothness (Lower is Better)')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Success Rate
    ax = axes[1, 0]
    success_rates = df.groupby('planner')['success'].mean() * 100
    ax.bar(range(len(success_rates)), success_rates.values)
    ax.set_xticks(range(len(success_rates)))
    ax.set_xticklabels(success_rates.index)
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. PPE Metric
    ax = axes[1, 1]
    data_ppe = df_success.groupby('planner')['ppe'].apply(list)
    ax.boxplot([data_ppe[p] for p in data_ppe.index], labels=data_ppe.index)
    ax.set_ylabel('PPE Metric')
    ax.set_title('Personalized Path Efficiency')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('benchmark_comparison.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: benchmark_comparison.png")
    plt.close()


if __name__ == '__main__':
    # Run benchmark with moderate parameters for demonstration
    df_results = run_benchmark(
        n_trials=10,  # Reduced for faster execution
        grid_sizes=[50],
        obstacle_densities=[0.2]
    )
