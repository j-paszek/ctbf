# CTBF 1.2.0 numerical evidence candidate

Read `bundle-manifest.json` for this directory's kind and publication status,
`SCHEMA.md` for column definitions, and `file-rights.tsv` for exact coverage.
Use the matching CTBF code release's `reproducibility/REPRODUCE.md` commands.
No reconstruction, simulation, external installation or raw-read download is
needed for stored-table export and numerical plotting.

The synthetic bundle contains original project numerical summaries and
selected per-history evidence under CC BY 4.0, plus original MIT
configuration/provenance metadata. It does not contain direct upstream tree
files, all simulated input banks, source snapshots or raw posterior traces.
It reproduces the reported numerical tables and plots from stored evidence,
not every stage of their original calculation.

The empirical bundle is a **LOCAL DRAFT pending a recorded rights review**.
It contains the retained CNP inputs, before/after-collapse profiles and
mappings, the CTBF feasibility report including its trees, empirical
diagnostic summaries, and the accepted editable SVG. Public availability is
not implied by preparing or successfully reproducing this draft.
Raw reads, BAMs, references, CopyKit objects, external software and MCMC
traces are excluded. Absolute copy-number scale remains unresolved and no
empirical accuracy against a known true tree is claimed.

Verify all source checksums using `python -m reproducibility.bundle verify
--bundle /path/to/this-directory` from the matching code checkout.
Historical local paths in immutable provenance are not read by this command.
Do not relocate files within the bundle or replace originals to make a check
pass. Use a new candidate directory after reviewing any deliberate change.
