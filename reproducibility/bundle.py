"""Prepare and verify allowlisted evidence copies, without mutating originals."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil

from .catalogue import SOURCES

REPO = Path(__file__).resolve().parents[1]
MANIFEST = "bundle-manifest.json"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_file(root, relative):
    """Do not follow any symlink, traverse parents, or consult original paths."""
    value = PurePosixPath(relative)
    if not value.parts or value.is_absolute() or ".." in value.parts or "\\" in relative:
        raise ValueError(f"Unsafe bundle path: {relative}")
    current = Path(root)
    if current.is_symlink():
        raise ValueError(f"Symlinked bundle root: {current}")
    for part in value.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"Symlinked input: {current}")
    if not current.is_file():
        raise ValueError(f"Missing regular input: {current}")
    return current


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


class Bundle:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads(safe_file(self.root, MANIFEST).read_text())
        if self.manifest.get("schema") != "ctbf-evidence-bundle-1":
            raise ValueError("Unrecognized bundle schema")
        self.files = self.manifest["files"]
        paths = [row["path"] for row in self.files]
        keys = [row["key"] for row in self.files]
        if len(paths) != len(set(paths)) or len(keys) != len(set(keys)):
            raise ValueError("Duplicate bundle paths/keys")
        for row in self.files:
            file = safe_file(self.root, row["path"])
            if sha256(file) != row["sha256"]:
                raise ValueError(f"Bundle checksum mismatch: {row['path']}")
        self.by_key = {row["key"]: row for row in self.files}
        empirical = self.manifest.get("kind") == "empirical"
        if self.manifest.get("kind") not in ("empirical", "synthetic"):
            raise ValueError("Missing bundle kind")
        required = {s.key: s for s in SOURCES if s.empirical == empirical}
        metadata = {"LICENSE", "LICENSESCOPE.md", "THIRD_PARTY_NOTICES.md", "README.md",
                    "SCHEMA.md", "evidence-map.tsv", "file-rights.tsv"}
        if not empirical:
            metadata |= {"LICENSE-DATA", "DATA_ATTRIBUTION.md"}
        if set(self.by_key) != set(required) | metadata:
            raise ValueError("Bundle source/metadata keys differ from the explicit allowlist")
        pins = json.loads((REPO / "reproducibility/source-pins.json").read_text())
        for key, source in required.items():
            row = self.by_key[key]
            if row["path"] != source.destination or row["sha256"] != pins[key]:
                raise ValueError(f"Bundle differs from reviewed source: {key}")
            if row["rights"] != source.rights:
                raise ValueError(f"Incorrect rights classification: {key}")
        actual_paths = set()
        for parent, dirs, names in os.walk(self.root, followlinks=False):
            for name in dirs + names:
                path = Path(parent) / name
                if path.is_symlink():
                    raise ValueError(f"Symlink in bundle: {path}")
            actual_paths.update((Path(parent) / name).relative_to(self.root).as_posix() for name in names)
        if actual_paths - set(paths) - {MANIFEST, "checksums.sha256"}:
            raise ValueError("Unlisted file in bundle; no automatic inclusion is allowed")
        with self.path("file-rights.tsv").open(newline="", encoding="utf-8") as stream:
            rights = list(csv.DictReader(stream, delimiter="\t"))
        if len(rights) != len({r["path"] for r in rights}):
            raise ValueError("Duplicate file-rights entries")
        expected = {row["path"]: row["rights"] for row in self.files}
        expected.update({MANIFEST: "MIT-documentation", "checksums.sha256": "MIT-documentation"})
        if {r["path"]: r["rights"] for r in rights} != expected:
            raise ValueError("file-rights.tsv does not cover the exact inventory")

    def path(self, key):
        return safe_file(self.root, self.by_key[key]["path"])

    def json(self, key):
        return json.loads(self.path(key).read_text())

    def csv(self, key):
        with self.path(key).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def check_publication(self, receipt=None):
        """A draft flag cannot be removed to bypass a file-specific rights review."""
        empirical = self.manifest["kind"] == "empirical"
        if not empirical:
            return
        if receipt is None:
            raise ValueError("Empirical bundle is LOCAL ONLY: recorded rights review required")
        review = json.loads(Path(receipt).read_text())
        if (review.get("schema") != "ctbf-empirical-rights-review-1"
                or review.get("decision") != "approved_for_public_distribution"
                or review.get("bundle_manifest_sha256") != sha256(self.root / MANIFEST)):
            raise ValueError("Empirical rights review is missing, unapproved, or belongs to different bytes")
        for field in ("reviewer", "reviewed_on", "source_rights_basis", "consent_and_privacy_basis",
                      "redistribution_terms"):
            if not isinstance(review.get(field), str) or not review[field].strip():
                raise ValueError(f"Incomplete empirical rights review: {field}")
        covered = review.get("covered_files", [])
        expected = {r["path"] for r in self.files if r["rights"] == "empirical-review-required"}
        if len(covered) != len(set(covered)) or set(covered) != expected:
            raise ValueError("Empirical review must cover every empirical file exactly once")


def prepare(ctbf, bio, output, empirical=False):
    """Explicit inputs only. Check every pin before making a new destination."""
    roots = {"ctbf": Path(ctbf), "ctbfbio": Path(bio)}
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError(f"Refusing to overwrite bundle: {output}")
    pins = json.loads((REPO / "reproducibility/source-pins.json").read_text())
    selected = [s for s in SOURCES if s.empirical == empirical]
    files = []
    for source in selected:
        path = safe_file(roots[source.origin], source.path)
        actual = sha256(path)
        if actual != pins[source.key]:
            raise ValueError(f"Reviewed input changed: {source.origin}/{source.path}")
        files.append((source, path, actual))
    output.mkdir(parents=True)
    entries = []

    def copy(source, relative, key, rights, origin="release-documentation"):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with Path(source).open("rb") as inp, target.open("xb") as out:
            shutil.copyfileobj(inp, out)
        entries.append(dict(key=key, path=relative, sha256=sha256(target), rights=rights,
                            original=origin))

    for source, original, digest in files:
        copy(original, source.destination, source.key, source.rights,
             f"{source.origin}:{source.path}")
        if entries[-1]["sha256"] != digest:
            raise ValueError("Source changed during copy; incomplete bundle preserved for review")
    for name in ("LICENSE", "LICENSESCOPE.md", "THIRD_PARTY_NOTICES.md"):
        root = roots["ctbfbio"] if empirical else roots["ctbf"]
        copy(safe_file(root, name), name, name, "MIT-notice")
    if not empirical:
        for name in ("LICENSE-DATA", "DATA_ATTRIBUTION.md"):
            copy(safe_file(roots["ctbf"], name), f"ctbf-generated/{name}", name, "licence-notice")
    for name, src in (("README.md", "BUNDLE_README.md"), ("SCHEMA.md", "SCHEMA.md"),
                      ("evidence-map.tsv", "evidence-map.tsv")):
        copy(REPO / "reproducibility" / src, name, name, "MIT-documentation")
    with (output / "file-rights.tsv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("path", "rights", "basis"))
        for row in entries:
            writer.writerow((row["path"], row["rights"], "LICENSESCOPE.md; THIRD_PARTY_NOTICES.md"))
        writer.writerow(("file-rights.tsv", "MIT-documentation", "Original inventory"))
        writer.writerow((MANIFEST, "MIT-documentation", "Original inventory"))
        writer.writerow(("checksums.sha256", "MIT-documentation", "Generated at packaging"))
    entries.append(dict(key="file-rights.tsv", path="file-rights.tsv",
                        sha256=sha256(output / "file-rights.tsv"), rights="MIT-documentation",
                        original="generated inventory"))
    write_json(output / MANIFEST, dict(schema="ctbf-evidence-bundle-1",
               kind="empirical" if empirical else "synthetic",
               publication_status="local_draft_pending_rights_review" if empirical else "synthetic_candidate",
               level="Stored numerical evidence and rendering; no new inference or bootstrap draws",
               files=entries))
    Bundle(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--ctbf-root", type=Path, default=REPO)
    p.add_argument("--bio-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--empirical-draft", action="store_true")
    p = sub.add_parser("verify")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--public", action="store_true")
    p.add_argument("--rights-review", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.ctbf_root, args.bio_root, args.output, args.empirical_draft)
        print("complete: verified local bundle; no archive or publication")
    else:
        bundle = Bundle(args.bundle)
        if args.public:
            bundle.check_publication(args.rights_review)
        print(f"complete: {len(bundle.files)} verified bundle files")


if __name__ == "__main__":
    main()
