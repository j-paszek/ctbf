"""
Master Script: NJ Benchmark Pipeline

This script runs the complete benchmarking and analysis pipeline:
1. Benchmark all NJ variants
2. Perform statistical analysis
3. Generate visualizations
"""

import sys
import os
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nj_benchmark import run_benchmark
from nj_analyzer import analyze_benchmark_results, generate_markdown_report
from nj_visualizer import create_all_visualizations


def run_full_pipeline(max_seeds=None, output_dir='results', figures_dir='figures',
                     config='data/config_for_pic.json', bedfile='data/pic.csv',
                     parallel_algorithms=False, max_workers=None, timestamp_dirs=True):
    """
    Run the complete NJ benchmark pipeline.
    
    Parameters
    ----------
    max_seeds : int or None
        Maximum number of seeds to test (None = all)
    output_dir : str
        Directory for output CSV files
    figures_dir : str
        Directory for output figures
    config : str
        Configuration file for simulation
    bedfile : str
        Bedfile for simulation
    parallel_algorithms : bool
        Run algorithms in parallel (default: False)
    max_workers : int or None
        Maximum number of parallel workers (default: number of CPUs)
    timestamp_dirs : bool
        Add timestamp to output directories (default: True)
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*80)
    print(" "*20 + "NJ BENCHMARK PIPELINE")
    print("="*80)
    print(f"Timestamp: {timestamp}")
    print(f"Max seeds: {max_seeds if max_seeds else 'ALL'}")
    print(f"Output directory: {output_dir}")
    print(f"Figures directory: {figures_dir}")
    print(f"Parallel processing: {parallel_algorithms}")
    if parallel_algorithms:
        print(f"Max workers: {max_workers if max_workers else 'auto'}")
    print(f"Timestamp directories: {timestamp_dirs}")
    print("="*80 + "\n")
    
    # Add timestamp to directories if requested
    if timestamp_dirs:
        output_dir = os.path.join(output_dir, timestamp)
        figures_dir = os.path.join(figures_dir, timestamp)
    
    # Define file paths
    benchmark_csv = os.path.join(output_dir, 'nj_benchmark_results.csv')
    summary_csv = os.path.join(output_dir, 'nj_summary_statistics.csv')
    tests_csv = os.path.join(output_dir, 'nj_statistical_tests.csv')
    roc_csv = os.path.join(output_dir, 'nj_roc_data.csv')
    report_md = os.path.join(output_dir, 'nj_analysis_report.md')
    
    # Step 1: Run benchmark
    print("\n" + "="*80)
    print("STEP 1: BENCHMARKING")
    print("="*80)
    
    try:
        df_benchmark = run_benchmark(
            output_csv=benchmark_csv,
            config=config,
            bedfile=bedfile,
            max_seeds=max_seeds,
            parallel_algorithms=parallel_algorithms,
            max_workers=max_workers,
            timestamp_dirs=False  # Already handled above
        )
        print(f"\n✓ Benchmark completed successfully")
        print(f"  Results saved to: {benchmark_csv}")
    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Analyze results
    print("\n" + "="*80)
    print("STEP 2: STATISTICAL ANALYSIS")
    print("="*80)
    
    try:
        summary_df, tests_df, roc_df, auc_df = analyze_benchmark_results(
            input_csv=benchmark_csv,
            use_input_dir=True  # Place outputs in same directory as benchmark_csv
        )
        print(f"\n✓ Analysis completed successfully")
        print(f"  Summary statistics: {summary_csv}")
        print(f"  Statistical tests: {tests_csv}")
        print(f"  ROC data: {roc_csv}")
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Generate markdown report
    print("\n" + "="*80)
    print("STEP 3: GENERATING REPORT")
    print("="*80)
    
    try:
        generate_markdown_report(summary_df, tests_df, auc_df, output_file=report_md)
        print(f"\n✓ Report generated successfully")
        print(f"  Report: {report_md}")
    except Exception as e:
        print(f"\n✗ Report generation failed: {e}")
        return False
    
    # Step 4: Generate visualizations
    print("\n" + "="*80)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("="*80)
    
    try:
        create_all_visualizations(
            benchmark_csv=benchmark_csv,
            output_dir=figures_dir,
            use_input_dir=False  # Use figures_dir as specified
        )
        print(f"\n✓ Visualizations generated successfully")
        print(f"  Figures directory: {figures_dir}")
    except Exception as e:
        print(f"\n✗ Visualization generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nOutput files:")
    print(f"  1. Benchmark results:      {benchmark_csv}")
    print(f"  2. Summary statistics:     {summary_csv}")
    print(f"  3. Statistical tests:      {tests_csv}")
    print(f"  4. ROC data:               {roc_csv}")
    print(f"  5. Analysis report:        {report_md}")
    print(f"  6. Figures directory:      {figures_dir}/")
    print("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run complete NJ benchmark pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline on all seeds
  python run_nj_benchmark_pipeline.py
  
  # Run on first 10 seeds (for testing)
  python run_nj_benchmark_pipeline.py --max-seeds 10
  
  # Specify custom output directories
  python run_nj_benchmark_pipeline.py --output-dir my_results --figures-dir my_figures
        """
    )
    
    parser.add_argument('--max-seeds', type=int, default=None,
                       help='Maximum number of seeds to test (default: all)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for CSV files (default: results)')
    parser.add_argument('--figures-dir', type=str, default='figures',
                       help='Output directory for figures (default: figures)')
    parser.add_argument('--config', type=str, default='data/config_for_pic.json',
                       help='Configuration file (default: data/config_for_pic.json)')
    parser.add_argument('--bedfile', type=str, default='data/pic.csv',
                       help='Bedfile (default: data/pic.csv)')
    parser.add_argument('--parallel', action='store_true',
                       help='Run algorithms in parallel')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of parallel workers (default: number of CPUs)')
    parser.add_argument('--no-timestamp', action='store_true',
                       help='Disable timestamp in output directories')
    
    args = parser.parse_args()
    
    # Run pipeline
    success = run_full_pipeline(
        max_seeds=args.max_seeds,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        config=args.config,
        bedfile=args.bedfile,
        parallel_algorithms=args.parallel,
        max_workers=args.max_workers,
        timestamp_dirs=not args.no_timestamp
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
