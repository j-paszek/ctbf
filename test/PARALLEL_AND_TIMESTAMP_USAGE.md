# Parallel Processing and Timestamp Features

## Overview

The NJ benchmarking suite now supports:
1. **Parallel algorithm processing** - Run multiple NJ variants simultaneously
2. **Timestamped output directories** - Organize results by run timestamp

## Features Added

### 1. Parallel Algorithm Processing

Run different NJ algorithms in parallel across multiple CPU cores to speed up benchmarking.

**Benefits:**
- Faster benchmarking when testing multiple algorithms
- Better CPU utilization
- Same output format as sequential processing

**How it works:**
- Each algorithm is run on all seeds in a separate process
- Seeds for each algorithm are still processed sequentially (for stability)
- Results are collected and merged into single CSV file

### 2. Timestamped Directories

Automatically organize results and figures by run timestamp for easy comparison between runs.

**Benefits:**
- Compare results from different runs
- Never overwrite previous results
- Easy to track when benchmarks were performed

**Directory structure:**
```
results/
├── 20251105_143022/          # Timestamp: YYYYMMDD_HHMMSS
│   ├── nj_benchmark_results.csv
│   ├── nj_summary_statistics.csv
│   ├── nj_statistical_tests.csv
│   ├── nj_roc_data.csv
│   └── nj_analysis_report.md
└── 20251105_150415/          # Another run
    └── ...

figures/
├── 20251105_143022/
│   ├── boxplot_multiset_f1.png
│   ├── boxplot_unique_f1.png
│   └── ... (11 total figures)
└── 20251105_150415/
    └── ...
```

---

## Usage

### Master Pipeline Script

#### Basic Usage (with timestamps)
```bash
cd /Users/krzysiek/PROJECTS-MIMUW/ctbf/test
python run_nj_benchmark_pipeline.py --max-seeds 5
```

This creates:
- `results/YYYYMMDD_HHMMSS/` with all CSV files
- `figures/YYYYMMDD_HHMMSS/` with all PNG files

#### With Parallel Processing
```bash
python run_nj_benchmark_pipeline.py --max-seeds 5 --parallel
```

- Runs algorithms in parallel (default: uses all CPU cores)
- Creates timestamped directories

#### Specify Number of Workers
```bash
python run_nj_benchmark_pipeline.py --max-seeds 5 --parallel --max-workers 4
```

- Uses exactly 4 parallel workers

#### Disable Timestamps (old behavior)
```bash
python run_nj_benchmark_pipeline.py --max-seeds 5 --no-timestamp
```

- Creates `results/` and `figures/` directly (no timestamp subdirectory)
- **Warning**: Will overwrite previous results!

#### Full Example with All Options
```bash
python run_nj_benchmark_pipeline.py \
  --max-seeds 10 \
  --parallel \
  --max-workers 8 \
  --output-dir my_results \
  --figures-dir my_figures
```

---

### Individual Scripts

#### nj_benchmark.py

**With timestamps (default):**
```bash
python nj_benchmark.py --max-seeds 5
# Output: results/YYYYMMDD_HHMMSS/nj_benchmark_results.csv
```

**With parallel processing:**
```bash
python nj_benchmark.py --max-seeds 5 --parallel
```

**With both:**
```bash
python nj_benchmark.py --max-seeds 5 --parallel --max-workers 4
```

**Disable timestamps:**
```bash
python nj_benchmark.py --max-seeds 5 --no-timestamp
# Output: results/nj_benchmark_results.csv
```

**All options:**
```bash
python nj_benchmark.py \
  --max-seeds 10 \
  --output results/nj_benchmark_results.csv \
  --parallel \
  --max-workers 4 \
  --no-timestamp
```

#### nj_analyzer.py

Automatically uses the same directory as input CSV:

```bash
# Analyze timestamped results
python nj_analyzer.py --input results/20251105_143022/nj_benchmark_results.csv

# Outputs created in results/20251105_143022/:
#   - nj_summary_statistics.csv
#   - nj_statistical_tests.csv
#   - nj_roc_data.csv
#   - nj_analysis_report.md
```

To place outputs elsewhere:
```bash
python nj_analyzer.py \
  --input results/20251105_143022/nj_benchmark_results.csv \
  --no-use-input-dir
```

#### nj_visualizer.py

Automatically uses the same parent directory as input CSV:

```bash
# Visualize timestamped results
python nj_visualizer.py --benchmark-csv results/20251105_143022/nj_benchmark_results.csv

# Outputs created in figures/20251105_143022/:
#   - 11 PNG files
```

To place figures elsewhere:
```bash
python nj_visualizer.py \
  --benchmark-csv results/20251105_143022/nj_benchmark_results.csv \
  --output-dir custom_figures \
  --no-use-input-dir
```

---

## Performance Considerations

### Parallel Processing

**When to use parallel:**
- Testing many algorithms (5+)
- Have multiple CPU cores available
- Seeds take significant time (5+ minutes each)

**When NOT to use parallel:**
- Testing only 1-2 algorithms (overhead not worth it)
- Limited memory (each worker uses memory)
- Debugging (sequential is easier to debug)

**Expected speedup:**
- With 8 cores and 15 algorithms: ~6-7x faster
- Actual speedup depends on CPU, memory, and simulation complexity
- Seeds within each algorithm run sequentially (most stable)

### Memory Usage

Each parallel worker runs a full simulation, so memory usage scales:
- Sequential: ~1GB per seed
- Parallel with N workers: ~N GB simultaneously

### Disk Space

Timestamps create separate directories for each run:
- Per run: ~500KB for 5 seeds, ~5MB for 50 seeds
- Figures: ~50MB per run (11 PNG files at 300 DPI)

---

## Comparing Results Between Runs

With timestamped directories, you can easily compare different runs:

```bash
# Run 1: Test with 5 seeds
python run_nj_benchmark_pipeline.py --max-seeds 5
# Creates: results/20251105_143022/

# Run 2: Test with 10 seeds (later)
python run_nj_benchmark_pipeline.py --max-seeds 10
# Creates: results/20251105_150415/

# Compare results
diff results/20251105_143022/nj_summary_statistics.csv \
     results/20251105_150415/nj_summary_statistics.csv

# Or load both in Python for analysis
import pandas as pd
run1 = pd.read_csv('results/20251105_143022/nj_benchmark_results.csv')
run2 = pd.read_csv('results/20251105_150415/nj_benchmark_results.csv')
```

---

## Default Behavior Summary

| Feature | Default | Override Flag |
|---------|---------|---------------|
| Timestamps | Enabled | `--no-timestamp` |
| Parallel processing | Disabled | `--parallel` |
| Max workers | Auto (all cores) | `--max-workers N` |
| Output consolidation | Auto (same dir as input) | `--no-use-input-dir` |

---

## Examples

### Quick Test (2 algorithms, 1 seed, no parallel)
```bash
# Edit nj_benchmark.py to uncomment only 2 algorithms
python run_nj_benchmark_pipeline.py --max-seeds 1
# Takes ~2 minutes, creates timestamped results
```

### Full Benchmark (all algorithms, 10 seeds, parallel)
```bash
python run_nj_benchmark_pipeline.py --max-seeds 10 --parallel
# Takes ~30-60 minutes with 8 cores
# Creates comprehensive timestamped results
```

### Production Run (all algorithms, all seeds, parallel)
```bash
python run_nj_benchmark_pipeline.py --parallel --max-workers 8
# Takes several hours
# Full benchmark with all available seeds
```

### Disable Timestamps (for automated testing)
```bash
python run_nj_benchmark_pipeline.py --max-seeds 1 --no-timestamp
# Always writes to results/ and figures/
# Good for CI/CD pipelines
```

---

## Troubleshooting

### Parallel Processing Issues

**Error: "pickle error" or "can't pickle function"**
- Some functions can't be serialized for parallel processing
- Try running without `--parallel` flag
- Check that all imported modules are available

**Error: Out of memory**
- Reduce `--max-workers` to use fewer parallel processes
- Run sequentially (remove `--parallel`)

**Progress bars not showing correctly**
- This is normal with parallel processing
- Check terminal for periodic status updates

### Timestamp Issues

**Can't find my results**
- Check for timestamped subdirectories in `results/` and `figures/`
- Use `ls -lt results/` to see most recent runs
- Look for directory names like `YYYYMMDD_HHMMSS`

**Want to use specific directory name**
- Use `--no-timestamp` flag
- Specify exact path with `--output-dir` and `--figures-dir`

---

*Last updated: 2025-11-05*
