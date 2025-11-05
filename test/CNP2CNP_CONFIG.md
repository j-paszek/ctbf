# CNP2CNP Configuration Guide

## Issue

The original code had hardcoded paths to the cnp2cnp tool:
```python
cnp2cnp_FOLDER = r"/Users/voronwe/Work/PyCharmProjects/cnp2cnp/examples"
cnp2cnp_FILE = r"/Users/voronwe/Work/PyCharmProjects/cnp2cnp/cnp2cnp.py"
```

This caused errors when running on different machines.

## Solution

### Option 1: Use Parallel Mode (Recommended for Benchmarking)

The benchmark scripts now use `parallel=True` by default, which computes distances using Python instead of the external cnp2cnp tool. This is:
- ✅ Faster (uses multiple CPU cores)
- ✅ No external dependencies
- ✅ Works on any machine

**No action needed** - the benchmark suite is already configured this way.

### Option 2: Configure cnp2cnp Paths (If Needed)

If you need to use the cnp2cnp tool (for `parallel=False` mode), you can:

#### A. Set Environment Variables

```bash
export CNP2CNP_FOLDER="/path/to/cnp2cnp/examples"
export CNP2CNP_FILE="/path/to/cnp2cnp/cnp2cnp.py"
python your_script.py
```

#### B. Place cnp2cnp in Project Directory

Create this structure:
```
ctbf/
├── cnp2cnp/
│   ├── cnp2cnp.py
│   └── examples/
├── ctbs.py
└── ...
```

The code will automatically find it.

#### C. Modify ctbs.py Directly

Edit lines 25-26 in `ctbs.py`:
```python
cnp2cnp_FOLDER = "/your/path/to/cnp2cnp/examples"
cnp2cnp_FILE = "/your/path/to/cnp2cnp/cnp2cnp.py"
```

## What Was Fixed

1. **ctbs.py** - Changed hardcoded paths to use:
   - Environment variables (if set)
   - Relative path in project directory (default)
   - Can be overridden per-call

2. **nj_benchmark.py** - Added `parallel=True` to avoid cnp2cnp dependency

## Testing

Test that the fix works:

```bash
cd test
python run_nj_benchmark_pipeline.py --max-seeds 1
```

Should run without the "No such file or directory" error.

## Technical Details

### Before (Line 22 in ctbs.py)
```python
cnp2cnp_FOLDER = r"/Users/voronwe/Work/PyCharmProjects/cnp2cnp/examples"
```

### After (Lines 22-28 in ctbs.py)
```python
# cnp2cnp paths - configurable via environment variables
_default_cnp2cnp_folder = os.path.join(os.path.dirname(__file__), "cnp2cnp", "examples")
_default_cnp2cnp_file = os.path.join(os.path.dirname(__file__), "cnp2cnp", "cnp2cnp.py")

cnp2cnp_FOLDER = os.environ.get("CNP2CNP_FOLDER", _default_cnp2cnp_folder)
cnp2cnp_FILE = os.environ.get("CNP2CNP_FILE", _default_cnp2cnp_file)
```

## When to Use parallel=True vs parallel=False

### parallel=True (Default in Benchmarks)
- ✅ Faster for large datasets
- ✅ No external tool needed
- ✅ Uses all CPU cores
- ✅ Pure Python implementation
- ❌ May use more memory

### parallel=False (Legacy)
- ✅ Lower memory usage
- ✅ May be more accurate (if cnp2cnp is trusted reference)
- ❌ Requires cnp2cnp tool installed
- ❌ Slower (single-threaded external process)
- ❌ Requires path configuration

## Summary

**For benchmarking**: No action needed! The scripts use `parallel=True`.

**For other uses**: If you need `parallel=False`, configure paths using one of the methods above.
