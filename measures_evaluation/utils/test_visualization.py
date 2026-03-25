#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced dm_compare.py with visualization capabilities.
"""

import subprocess
import sys
from pathlib import Path

def test_dm_compare_with_visualization():
    """Test the dm_compare.py script with visualization enabled."""
    
    # Check if the distance matrix files exist
    file1 = Path("cnp_distance_matrix.txt")
    file2 = Path("sim_dm.txt")
    
    if not file1.exists():
        print(f"Warning: {file1} not found. Looking for alternative files...")
        # Try alternative files
        alt_files = [
            Path("sel/tests/dm.txt"),
            Path("sel/sel/simulator/distance_matrix.txt"), 
            Path("sel/sel/simulator/dm1.txt")
        ]
        
        available_files = [f for f in alt_files if f.exists()]
        if len(available_files) >= 2:
            file1, file2 = available_files[:2]
            print(f"Using {file1} and {file2}")
        else:
            print("Not enough distance matrix files found for comparison.")
            return False
    
    # Test without visualization first
    print("=== Testing dm_compare.py without visualization ===")
    cmd_basic = [
        sys.executable, "dm_compare.py", 
        str(file1), str(file2),
        "--permutations", "99"  # Fewer permutations for faster testing
    ]
    
    try:
        result = subprocess.run(cmd_basic, capture_output=True, text=True, check=True)
        print("Basic comparison successful!")
        print("Output preview:")
        print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Basic comparison failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    
    # Test with visualization
    print("\n=== Testing dm_compare.py with visualization ===")
    cmd_viz = [
        sys.executable, "dm_compare.py",
        str(file1), str(file2),
        "--visualize",
        "--permutations", "99",
        "--output-dir", "test_plots",
        "--plot-prefix", "test_comparison"
    ]
    
    try:
        result = subprocess.run(cmd_viz, capture_output=True, text=True, check=True)
        print("Visualization comparison successful!")
        print("Output preview:")
        print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        
        # Check if plot files were created
        plot_dir = Path("test_plots")
        if plot_dir.exists():
            plot_files = list(plot_dir.glob("test_comparison_*.png"))
            print(f"\nGenerated {len(plot_files)} plot files:")
            for pf in plot_files:
                print(f"  - {pf}")
        
    except subprocess.CalledProcessError as e:
        print(f"Visualization comparison failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        
        # Check if it's due to missing plotting libraries
        if "matplotlib" in e.stderr or "seaborn" in e.stderr:
            print("\nNote: Visualization requires matplotlib and seaborn.")
            print("Install with: pip install matplotlib seaborn scipy")
        return False
    
    return True

def show_help():
    """Show help for the enhanced dm_compare.py script."""
    print("\n=== dm_compare.py Usage Examples ===")
    print("Basic comparison:")
    print("  python dm_compare.py file1.txt file2.txt")
    print()
    print("With visualization:")
    print("  python dm_compare.py file1.txt file2.txt --visualize")
    print()
    print("Custom output directory and prefix:")
    print("  python dm_compare.py file1.txt file2.txt --visualize --output-dir plots --plot-prefix my_analysis")
    print()
    print("Full options:")
    print("  python dm_compare.py file1.txt file2.txt \\")
    print("    --visualize \\")
    print("    --permutations 9999 \\")
    print("    --mantel pearson \\")
    print("    --output-dir analysis_plots \\")
    print("    --plot-prefix experiment_2024")

if __name__ == "__main__":
    print("Testing enhanced dm_compare.py with visualization features")
    print("=" * 60)
    
    show_help()
    
    success = test_dm_compare_with_visualization()
    
    if success:
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
