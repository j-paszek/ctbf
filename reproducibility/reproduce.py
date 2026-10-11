"""Portable, read-only stored-result export and rendering for release 1.2.0."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess

from .bundle import Bundle, REPO, MANIFEST, sha256, write_json
from .tables import export_tables, synthetic_tables, empirical_tables


def primary_plot(bundle, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = bundle.json("v1_summary")["registered_primary"]
    values = sorted(float(r["block_effect"]) for r in bundle.csv("primary_blocks"))
    if len(values) != 100 or min(values) <= 0:
        raise ValueError("Unexpected primary block effects")
    with plt.rc_context({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11,
                         'legend.fontsize': 10.5, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        fig, axis = plt.subplots(figsize=(6, 4), layout="constrained")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D8DEE5", linewidth=.65)
        axis.scatter(range(1, 101), values, s=18, color="#263B70", zorder=3)
        axis.axhline(0, color="#7A1F1F", linestyle="--", linewidth=1)
        ci = primary["bootstrap_interval"]
        axis.axhspan(ci["lower"], ci["upper"], color="#B8DED9", alpha=.8)
        axis.axhline(primary["block_effect_summary"]["mean"], color="#087F8C", linewidth=1.6)
        axis.set(xlabel="History blocks, ordered by effect", ylabel=r"Mean $M_2-M_0$ rooted-GRF effect", xlim=(0, 101))
        axis.set_title("Prespecified paired effect: 100/100 positive blocks", loc="left")
        for suffix in ("pdf", "svg", "png"):
            kwargs = {"metadata": {"CreationDate": None, "ModDate": None}} if suffix == "pdf" else {}
            fig.savefig(output / f"primary-block-effects.{suffix}", dpi=220, facecolor="white", **kwargs)
        plt.close(fig)


def render_figures(bundle, output):
    # Call only the established renderer. Its original-workspace loader/audit
    # is deliberately not called or weakened to accommodate relocated inputs.
    from paper.figures.build_confirmation_figure_v6 import render
    fixed = bundle.csv("v3_counts")
    render(output, native=bundle.csv("native_counts"), partial_trim=bundle.csv("trimmed_counts"),
           closed_trim=bundle.csv("closed_trimmed"), fixed=fixed,
           partial_full=fixed + bundle.csv("m4_counts"), matched=bundle.csv("v5_means"),
           height_ad=bundle.csv("v1_means"), height_grf=bundle.csv("v4_means"))
    primary_plot(bundle, output)


def render_vectors(output, empirical=None):
    for name in ("connect-then-complete-owner-v2", "nonidentifiability-v2", "trimmed-true-tree-v1"):
        source = REPO / "paper/figures" / (name + ".tex")
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                        "-outdir=" + str(output.resolve()), str(source)], check=True, cwd=output)
    if empirical is not None:
        subprocess.run(["inkscape", str(empirical.path("empirical_svg")),
                        "--export-filename=" + str((output / "ktn152-empirical-v4.pdf").resolve())], check=True)


def run(bundle_path, output, empirical_path=None, figures=False, vectors=False, supplement=None):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError(f"Refusing to overwrite output: {output}")
    bundle = Bundle(bundle_path)
    if bundle.manifest["kind"] != "synthetic":
        raise ValueError("--bundle must name the synthetic bundle")
    empirical = Bundle(empirical_path) if empirical_path is not None else None
    if empirical is not None and empirical.manifest["kind"] != "empirical":
        raise ValueError("--empirical-bundle must name the empirical bundle")
    tables = synthetic_tables(bundle)
    if empirical is not None:
        tables.update(empirical_tables(empirical))
    if supplement is not None:
        from .manuscript_audit import audit
        checked = audit(supplement, tables)
        print(f"verified: {checked} printed decimal values across {len(tables)} tables")
    output.mkdir(parents=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output.resolve() / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output.resolve() / ".cache"))
    export_tables(output / "tables", tables)
    # Formal tables are source mappings, not invented experimental records.
    for key in ("method_identities", "simulation"):
        shutil.copyfile(bundle.path(key), output / (key + ".json"))
    shutil.copyfile(REPO / "reproducibility/method-map.tsv", output / "method-map.tsv")
    if figures:
        target = output / "figures"
        target.mkdir()
        render_figures(bundle, target)
    if vectors:
        target = output / "vectors"
        target.mkdir()
        render_vectors(target, empirical)
    manifests = {"synthetic": sha256(Path(bundle_path) / MANIFEST)}
    if empirical:
        manifests["empirical_local_draft"] = sha256(Path(empirical_path) / MANIFEST)
    write_json(output / "reproduction.json", dict(schema="ctbf-paper-export-1",
        input_manifests=manifests, numerical_tables=len(tables),
        table_rows={key: len(rows) for key, rows in tables.items()},
        figures=figures, vectors=vectors, new_bootstrap_draws=0, new_inferences=0,
        empirical_publication_review="not implied by local reproduction"))
    print(f"complete: {len(tables)} numerical tables; stored intervals preserved; no inference")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--empirical-bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", action="store_true", help="Render the three numerical figures")
    parser.add_argument("--vectors", action="store_true", help="Also compile schematics and, if supplied, empirical SVG; needs latexmk/Inkscape")
    parser.add_argument("--check-supplement", type=Path, help="Optional author-workspace cross-check, never a numerical source")
    args = parser.parse_args()
    run(args.bundle, args.output, args.empirical_bundle, args.figures, args.vectors, args.check_supplement)


if __name__ == "__main__":
    main()
