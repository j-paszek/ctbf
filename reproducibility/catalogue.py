"""Explicit release inputs. No recursive source discovery or upstream software."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    key: str
    origin: str
    path: str
    empirical: bool = False

    @property
    def rights(self):
        if self.empirical:
            return "empirical-review-required"
        return "MIT" if self.key in ("simulation", "design", "scientific_lock") else "CC-BY-4.0"

    @property
    def destination(self):
        scope = "empirical-draft" if self.empirical else ("original-metadata" if self.rights == "MIT" else "ctbf-generated")
        return f"{scope}/{self.key}/{self.path.rsplit('/', 1)[-1]}"


SOURCES = (
    Source("v1_summary", "ctbf", "paper/figures/v5-confirmation-v1/evidence-summary.json"),
    Source("v1_means", "ctbf", "paper/figures/v5-confirmation-v1/main-figure-data.csv"),
    Source("v1_contrasts", "ctbf", "paper/figures/v5-confirmation-v1/contrast-by-height.csv"),
    Source("primary_blocks", "ctbf", "paper/figures/v5-confirmation-v1/primary-block-effects.csv"),
    Source("v3_counts", "ctbf", "paper/figures/v5-confirmation-v3/biopsy-count-curves.csv"),
    Source("v3_blocks", "ctbf", "paper/figures/v5-confirmation-v3/biopsy-count-blocks.csv"),
    Source("method_identities", "ctbf", "paper/figures/v5-confirmation-v3/method-identities.json"),
    Source("v3_audit", "ctbf", "paper/figures/v5-confirmation-v3/audit.json"),
    Source("v4_means", "ctbf", "paper/figures/v5-confirmation-v4/closed-state-grf-curves.csv"),
    Source("v4_contrasts", "ctbf", "paper/figures/v5-confirmation-v4/closed-state-grf-contrasts.csv"),
    Source("v4_blocks", "ctbf", "paper/figures/v5-confirmation-v4/closed-state-grf-blocks.csv"),
    Source("v5_means", "ctbf", "paper/figures/v5-confirmation-v5/partial-matched-curves.csv"),
    Source("m4_counts", "ctbf", "paper/figures/v5-confirmation-v5/m4-biopsy-count-curves.csv"),
    Source("m4_placement", "ctbf", "paper/review/sections/supplementary-methods/iteration-24/evidence/summary.json"),
    Source("m4_placement_pairs", "ctbf", "paper/review/sections/supplementary-methods/iteration-24/evidence/paired-differences.csv"),
    Source("v5_audit", "ctbf", "paper/figures/v5-confirmation-v5/audit.json"),
    Source("native_counts", "ctbf", "paper/figures/v5-confirmation-v6/native-biopsy-count-curves.csv"),
    Source("trimmed_counts", "ctbf", "paper/figures/v5-confirmation-v6/partial-trimmed-biopsy-count-curves.csv"),
    Source("closed_trimmed", "ctbf", "paper/figures/v5-confirmation-v6/closed-trimmed-height-curves.csv"),
    Source("v6_audit", "ctbf", "paper/figures/v5-confirmation-v6/audit.json"),
    Source("extension_report", "ctbf", "algorithm_evaluation/results/v5_nestedbd_long/partial_comparators_100_2m_v1/reports/report_001/report.json"),
    Source("resources", "ctbf", "algorithm_evaluation/results/v5_confirmation/condition_specific_resource_v1/summary.json"),
    Source("simulation", "ctbf", "simulator_examples/paper_v5/confirmatory_standard.json"),
    Source("design", "ctbf", "experimental_description/g2_01a_v5_confirmatory_manifest.json"),
    Source("scientific_lock", "ctbf", "experimental_description/g2_01a_v5_source_lock.json"),
    Source("empirical_report", "ctbfbio", "data/processed/ktn152_ctbf_feasibility_v1/report.json", True),
    Source("empirical_diagnostics", "ctbfbio", "data/processed/ktn152_nestedbd_long_corrected_review_v1/report.json", True),
    Source("empirical_svg", "ctbf", "paper/figures/ktn152-empirical-v4.svg", True),
) + tuple(
    Source("empirical_" + name.split(".")[0], "ctbfbio",
           "data/processed/ktn152_copykit_feasibility_v1/" + name, True)
    for name in ("analysis_contract.json", "bins.tsv", "cell_disposition.tsv",
                 "ctbf_input.json", "duplicate_members.tsv", "export_summary.json",
                 "profiles_collapsed.tsv", "profiles_uncollapsed.tsv",
                 "source_files.tsv", "data_manifest.tsv")
)
