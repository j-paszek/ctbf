# Parallelization Optimization

## Summary of Changes

The benchmarking parallelization has been significantly improved for better performance and resource efficiency.

## Key Improvements

### 1. **Flattened Task Structure**
**Before:** 
- Two nested loops: algorithms × seeds
- Could only parallelize across algorithms OR keep everything sequential
- Limited parallelization: max 15 workers (one per algorithm)

**After:**
- Single flat list of (algorithm, seed) tuples
- All tasks processed in parallel regardless of algorithm count
- Better utilization: 30 tasks (15 algorithms × 2 seeds) run concurrently with 7 workers

### 2. **Default Worker Limit: 60% of CPUs**
**Rationale:**
- Running at 100% CPU is expensive and can slow down the entire system
- 60% provides good parallelization while leaving resources for other tasks
- User explicitly requested this default

**Example:**
- 12 CPU cores → 7 workers (default)
- Can override with `--max-workers N` flag

### 3. **Simplified API**
**Parameter Renamed:**
- `parallel_algorithms=True/False` → `parallel=True/False`
- More intuitive: enables parallel processing of all tasks, not just algorithms

### 4. **Results Sorting**
- Results are sorted by (method, seed) after collection
- Makes CSV more organized even though tasks complete in random order

## Performance Comparison

### Sequential Mode (--parallel flag NOT used)
```bash
python nj_benchmark.py --max-seeds 5
```
- Processes: 15 algorithms × 5 seeds = 75 tasks
- Time estimate: ~75 × 12s = 15 minutes
- Resource usage: 1 CPU core at a time

### Parallel Mode (--parallel flag used)
```bash
python nj_benchmark.py --max-seeds 5 --parallel
```
- Processes: 75 tasks with 7 workers
- Time estimate: ~75/7 × 12s = ~2.2 minutes (6.8x faster)
- Resource usage: 7 CPU cores (60% of 12)

### Custom Worker Count
```bash
python nj_benchmark.py --max-seeds 5 --parallel --max-workers 4
```
- Uses exactly 4 workers regardless of CPU count

## Usage Examples

### Basic parallel benchmarking:
```bash
python nj_benchmark.py --max-seeds 10 --parallel
```

### Full pipeline with parallel processing:
```bash
python run_nj_benchmark_pipeline.py --max-seeds 20 --parallel
```

### Conservative parallel mode (fewer workers):
```bash
python nj_benchmark.py --max-seeds 10 --parallel --max-workers 4
```

### Aggressive parallel mode (more workers):
```bash
python nj_benchmark.py --max-seeds 10 --parallel --max-workers 10
```

## Technical Details

### Implementation
- Uses `ProcessPoolExecutor` from `concurrent.futures`
- Each worker processes one (algorithm, seed) task at a time
- Tasks are distributed automatically by the executor
- Progress bar shows real-time completion across all workers

### Task Function
```python
def evaluate_single_task(task_tuple):
    """
    Evaluate a single (algorithm, seed) combination.
    
    Parameters: (algo_name, algo_func, seed, config, bedfile, 
                 biopsy_size_scalable, biopsy_generations, r_dist)
    
    Returns: [nj_pure_result, ctbs_hybrid_result]
    """
```

### Why This is Better
1. **Maximum Utilization**: Uses available cores efficiently
2. **Scalability**: Works well with any number of algorithms and seeds
3. **Cost-Effective**: 60% default prevents excessive resource usage
4. **Flexibility**: Easy to adjust worker count based on needs
5. **Progress Tracking**: Single progress bar for all tasks

## Migration Notes

If you have scripts using the old API:
- Change `parallel_algorithms=True` → `parallel=True`
- Default worker count changed from 100% to 60% of CPUs
- No other changes needed

## Recommendations

- **For development/testing**: Use `--max-seeds 5` without `--parallel`
- **For moderate runs**: Use `--max-seeds 20 --parallel` (default 60% workers)
- **For production runs**: Use `--max-seeds 100 --parallel --max-workers 10` (adjust based on available resources)
- **For cost-sensitive environments**: Use fewer workers with `--max-workers 4`
