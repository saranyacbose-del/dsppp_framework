# Dynamic Semantic Personalized Path Planning (DSPPP) Framework

## Overview
This is a Python implementation of the DSPPP framework for autonomous navigation with dynamic obstacle prediction, semantic awareness, and personalized routing.

## Features
- Improved A* algorithm with dynamic penalty costs
- Kalman filter-based obstacle trajectory prediction
- Semantic-aware routing using OpenStreetMap data
- Inverse Reinforcement Learning for user preference learning
- Real-time replanning at 10 Hz update rates
- Comprehensive evaluation metrics including PPE (Personalized Path Efficiency)

## Requirements
- Python 3.8+
- NumPy 1.24+
- SciPy 1.11+
- NetworkX 3.1+
- OSMnx 1.8+
- Matplotlib 3.7+
- Seaborn 0.12+

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Run on Synthetic Grid Environment
```python
python examples/run_grid_experiment.py
```

### Run on Real OSM Network
```python
python examples/run_osm_experiment.py
```

### Run Complete Benchmark
```python
python examples/run_benchmark.py
```

## Project Structure

```
dsppp_framework/
├── algorithms/          # Path planning algorithms
│   ├── improved_astar.py
│   ├── baseline_planners.py
│   └── dynamic_penalty.py
├── environment/         # Environment representations
│   ├── grid_env.py
│   └── osm_env.py
├── semantic/            # Semantic processing
│   ├── feature_extraction.py
│   └── preference_learning.py
├── utils/              # Utilities
│   ├── metrics.py
│   ├── visualization.py
│   └── statistics.py
├── experiments/        # Experiment runners
│   └── test_harness.py
├── examples/           # Example scripts
└── config/            # Configuration files
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{saranya2025dsppp,
  title={Semantic-Aware Real-Time Path Planning for Autonomous Navigation with Dynamic Obstacle Prediction},
  author={Saranya, C and Janaki, G},
  journal={Your Journal},
  year={2025}
}
```

## License
MIT License

## Authors
- Saranya C (saranyaresearch22@gmail.com)
- Janaki G (janakig@srmist.edu.in)
