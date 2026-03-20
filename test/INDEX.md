# 📚 NJ Benchmark Suite - Documentation Index

Welcome to the NJ Benchmark Suite documentation! This index will help you find what you need quickly.

## 🎯 I Want To...

### Get Started Quickly
👉 **[QUICKSTART.md](QUICKSTART.md)** - 5-minute guide to running your first benchmark

### Understand Everything
👉 **[README_NJ_BENCHMARK.md](README_NJ_BENCHMARK.md)** - Complete documentation (200+ lines)

### See What's Available
👉 **[BENCHMARK_SUITE_SUMMARY.md](BENCHMARK_SUITE_SUMMARY.md)** - Full feature list and capabilities

### Run the Pipeline
👉 Run: `python run_nj_benchmark_pipeline.py --max-seeds 5`

### Explore Interactively
👉 Open: `jupyter notebook nj_benchmark_analysis.ipynb`

### Get Quick Insights
👉 Run: `python nj_utils.py summary`

## 📂 File Reference

### Scripts (What to Run)
| File | Purpose | Usage |
|------|---------|-------|
| `run_nj_benchmark_pipeline.py` | **Run everything** | `python run_nj_benchmark_pipeline.py` |
| `nj_benchmark.py` | Run benchmarks only | `python nj_benchmark.py --max-seeds 10` |
| `nj_analyzer.py` | Analyze existing results | `python nj_analyzer.py` |
| `nj_visualizer.py` | Generate figures | `python nj_visualizer.py` |
| `nj_utils.py` | Quick utilities | `python nj_utils.py summary` |

### Notebooks (Interactive)
| File | Purpose |
|------|---------|
| `nj_benchmark_analysis.ipynb` | Interactive analysis with examples |

### Documentation (What to Read)
| File | Best For |
|------|----------|
| `QUICKSTART.md` | Getting started (5 min) |
| `README_NJ_BENCHMARK.md` | Complete reference |
| `BENCHMARK_SUITE_SUMMARY.md` | Feature overview |
| `INDEX.md` | This file - navigation |

### Configuration
| File | Purpose |
|------|---------|
| `requirements_nj_benchmark.txt` | Python dependencies |

## 🎓 Learning Path

### Beginner
1. Read [QUICKSTART.md](QUICKSTART.md)
2. Install: `pip install -r requirements_nj_benchmark.txt`
3. Run test: `python run_nj_benchmark_pipeline.py --max-seeds 5`
4. View summary: `python nj_utils.py summary`

### Intermediate
1. Read [README_NJ_BENCHMARK.md](README_NJ_BENCHMARK.md)
2. Run full benchmark: `python run_nj_benchmark_pipeline.py`
3. Explore results: `jupyter notebook nj_benchmark_analysis.ipynb`
4. Try utilities: `python nj_utils.py compare ctbs hybrid_opt`

### Advanced
1. Read [BENCHMARK_SUITE_SUMMARY.md](BENCHMARK_SUITE_SUMMARY.md)
2. Customize methods in `nj_benchmark.py`
3. Add custom visualizations to `nj_visualizer.py`
4. Write custom analysis scripts using the CSV outputs

## 📊 Output Reference

### Generated Directories
```
test/
├── results/         # CSV files and reports
└── figures/         # PNG visualizations
```

### Key Output Files
| File | Contains |
|------|----------|
| `results/nj_benchmark_results.csv` | Raw benchmark data |
| `results/nj_summary_statistics.csv` | Aggregated statistics |
| `results/nj_statistical_tests.csv` | P-values and significance |
| `results/nj_roc_data.csv` | ROC curve data |
| `results/nj_analysis_report.md` | Formatted report |
| `figures/*.png` | Publication-quality figures |

## 🔍 Quick Command Reference

```bash
# Installation
pip install -r requirements_nj_benchmark.txt

# Quick test (5 min)
python run_nj_benchmark_pipeline.py --max-seeds 5

# Full pipeline
python run_nj_benchmark_pipeline.py

# View summary
python nj_utils.py summary

# Compare methods
python nj_utils.py compare ctbs hybrid_opt

# Find best method
python nj_utils.py best multiset_f1

# Check consistency
python nj_utils.py consistency

# Analyze specific seed
python nj_utils.py seed 295

# Export table
python nj_utils.py export 10

# Interactive analysis
jupyter notebook nj_benchmark_analysis.ipynb
```

## 🎨 Figure Reference

Generated figures in `figures/`:

1. **boxplot_multiset_f1.png** - Distribution comparison (main figure for paper)
2. **boxplot_unique_f1.png** - Unique F1 distribution
3. **violin_multiset_f1.png** - Detailed distribution visualization
4. **roc_curves.png** - ROC/AUC comparison
5. **pvalue_heatmap.png** - Statistical significance (use in paper)
6. **performance_scatter.png** - Multiset vs Unique comparison
7. **method_ranking.png** - Performance ranking (use in paper)
8. **multi_metric_comparison.png** - All metrics side-by-side

## 📖 Documentation Sections

### In README_NJ_BENCHMARK.md
- Installation instructions
- Detailed script descriptions
- Metrics explanation
- Statistical tests explanation
- Usage examples
- Troubleshooting
- Advanced customization

### In QUICKSTART.md
- One-page overview
- Quick commands
- Output structure
- Interpretation guide
- Next steps

### In BENCHMARK_SUITE_SUMMARY.md
- Complete feature list
- Design decisions
- Extensibility guide
- Quality features
- What you get

## 🆘 Troubleshooting Quick Links

| Issue | Solution Location |
|-------|-------------------|
| Import errors | README_NJ_BENCHMARK.md → Troubleshooting |
| Slow performance | QUICKSTART.md → Use `--max-seeds` |
| Understanding metrics | README_NJ_BENCHMARK.md → Metrics Explained |
| Customizing plots | README_NJ_BENCHMARK.md → Advanced Usage |
| Statistical tests | README_NJ_BENCHMARK.md → Interpreting Results |

## 🎯 For Your Paper

### Figures to Include
1. `boxplot_multiset_f1.png` - Main comparison
2. `method_ranking.png` - Performance ranking
3. `pvalue_heatmap.png` - Statistical significance

### Tables to Include
1. Summary statistics from `nj_summary_statistics.csv`
2. Statistical tests from `nj_statistical_tests.csv`

### Text to Include
- Methods section: See README_NJ_BENCHMARK.md → Citation
- Results section: See `nj_analysis_report.md`
- Statistics: P-values from `nj_statistical_tests.csv`

## 🔗 External Resources

- **SciPy Documentation**: Statistical tests - https://docs.scipy.org/doc/scipy/reference/stats.html
- **Matplotlib Gallery**: Plot examples - https://matplotlib.org/stable/gallery/
- **Seaborn Tutorial**: Statistical plots - https://seaborn.pydata.org/tutorial.html
- **Pandas Documentation**: Data analysis - https://pandas.pydata.org/docs/

## 📝 Quick Notes

- All scripts run from `test/` directory
- Requires Python 3.7+
- Uses your existing simulator, reconstructor, evaluator
- No modifications to core codebase needed
- All outputs are CSV/PNG (easy to use)
- Fully documented and commented

## 🎉 You're All Set!

**Next step**: Open [QUICKSTART.md](QUICKSTART.md) and run your first benchmark!

Questions? Check the relevant documentation file above or examine the inline comments in the scripts.

---

*Created: November 2025*  
*Version: 1.0.0*  
*Compatible with: CTBF codebase*
