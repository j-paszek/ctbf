# Evidence and export schema

`bundle-manifest.json` (schema `ctbf-evidence-bundle-1`) records each copied
file's unique key, relative path, SHA-256, rights class and original logical
location (`ctbf:` or `ctbfbio:`). `source-pins.json` in the code release binds
the reviewed source bytes. Original files are copied without rewriting their
historical absolute paths. Portable readers never resolve those paths.
`file-rights.tsv` covers the inventory and its generated metadata. There are
no upstream code files, source snapshots, executables or raw MCMC traces.

`reproduce` writes one `tab-supp-*.csv` per numerical table, in long form,
and the same rows in `tables.json`. These are numerical exports, not a copy of
the manuscript's TeX layout. Columns absent from a particular row are empty,
not zero. `method`, `panel`, `condition` and `metric` identify the quantity.
`source_key` identifies its stored evidence. `paper_decimal_places` records
display precision; the CSV preserves full available numerical precision.

- `count` is the number of successful independent histories, except where
  explicitly marked as individual MCMC runs or pipeline inputs.
- `mean`, `median`, `lower`, `upper` are stored scores/effects and stored
  two-sided 95% bootstrap limits. The exporter performs no resampling.
- `wins`, `ties`, `losses` count signed history-level differences, not
  independent repetitions of height-specific conditions. The tolerance in
  the original summaries is retained. Larger similarity/AD-F1 is better.
- `grf_full` and `grf_trim` distinguish full and state-trimmed truth.
  `fixed_U2_ad_f1` fixes the scoring labels at the two-biopsy set.
  Native `ad_f1` scores all labels observed in the relevant condition.
- Resource tables use seconds and decimal MB for the synthetic audit,
  but seconds and binary MiB for the empirical table, matching the paper.
- ESS values summarize individual cold-chain runs; between-run clade
  differences compare two runs of the same input. They do not establish
  convergence. The diagnostic burn-in windows do not change scored trees.

`method-map.tsv` includes the eight principal CTBF methods, two comparators
and two secondary definitions. The preserved historical
`method_identities.json` describes the original seven-arm campaign, before
the additive M4 extension. M3 is only a display alias for `Pooled-Q`.

The per-history CSVs and JSON block arrays retain their original schemas and
identifiers. They support auditing, but this lightweight command does not
recompute evaluator scores, bootstrap draws, ESS or clades from raw trees.
Original execution paths in those files are provenance, not dependencies.

## Empirical publication review

Local preparation is permitted without public redistribution approval.
A separate owner review JSON must use schema `ctbf-empirical-rights-review-1`,
decision `approved_for_public_distribution`, and the exact
`bundle_manifest_sha256`. It must state `reviewer`, `reviewed_on`,
`source_rights_basis`, `consent_and_privacy_basis`, `redistribution_terms`,
and `covered_files` listing every path classified
`empirical-review-required`, exactly once. Save it outside the bundle as
`audit/empirical-rights-review.json` in the release workspace.

This is a recorded human review, not automated legal clearance. No default
approval is supplied. A missing, incomplete or stale review blocks steps 9,
12 and 13 before packaging/upload/publication. Do not edit the inventory to
remove the gate or assign the synthetic-data licence to empirical material.
If cleared, step 9 copies the review to `CTBFBIO-RIGHTS-REVIEW.json`, distributed
and checksummed alongside the empirical archive. The archive's draft status
records its preparation state; the reviewed companion notice supplies the
subsequent redistribution basis without rewriting the pinned inventory.
