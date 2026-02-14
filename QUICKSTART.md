# DSPPP Framework - Quick Start Guide

## Installation

1. Extract the ZIP file
2. Navigate to the directory:
```bash
cd dsppp_framework
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Examples

### 1. Basic Demo (Recommended for first run)
```bash
python examples/run_demo.py
```

This will:
- Create a 50x50 grid with obstacles
- Run DSPPP, Standard A*, and Dijkstra
- Compare performance metrics
- Generate visualization (output_dsppp_demo.png)

### 2. Comprehensive Benchmark
```bash
python examples/run_benchmark.py
```

This will:
- Run 10 trials with 5 different algorithms
- Generate statistical analysis
- Save results to benchmark_results.csv
- Create comparison plots

## Understanding the Output

### Key Metrics:
- **Computation Time**: How fast the algorithm runs (ms)
- **Path Length**: Total distance traveled
- **Path Smoothness**: Angular deviation (degrees, lower = smoother)
- **Safety Margin**: Minimum distance from obstacles
- **PPE**: Personalized Path Efficiency (higher = better overall)

### Expected Results (DSPPP vs Standard A*):
- ✓ ~43% faster computation
- ✓ ~34% smoother paths
- ✓ ~47% higher PPE metric
- ✓ Better safety margins

## Customization

Edit `config/default_config.py` to adjust:
- Grid size and obstacle density
- Algorithm parameters
- Penalty weights
- Number of trials

## File Structure

```
dsppp_framework/
├── algorithms/          # Core planning algorithms
├── environment/         # Grid and map environments
├── utils/              # Metrics and visualization
├── examples/           # Runnable demos
├── config/             # Configuration files
└── README.md           # Full documentation
```

## Troubleshooting

**Import Error**: Make sure you're running from the dsppp_framework directory

**No output file**: Check that matplotlib is properly installed

**Slow execution**: Reduce grid size or number of trials in examples/run_benchmark.py

## Citation

```bibtex
@article{saranya2025dsppp,
  title={Semantic-Aware Real-Time Path Planning for Autonomous Navigation 
         with Dynamic Obstacle Prediction},
  author={Saranya, C and Janaki, G},
  year={2025}
}
```

## Support

For questions or issues:
- Email: saranyaresearch22@gmail.com
- Review the paper for theoretical background
- Check examples/ folder for more usage patterns
