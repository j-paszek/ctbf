"""Optional author-workspace check: compare exported values with printed tables.

The manuscript is a comparison target, never an input to numerical exports.
Decimal multisets detect omitted/changed values and their multiplicities.
Method/condition coverage is checked separately by tables.validate_tables.
"""
from collections import Counter
from pathlib import Path
import re


def printed_values(name, rows):
    result = []
    for row in rows:
        digits = row.get("paper_decimal_places", 6)
        if name == "confirmatory-resources":
            fields = [(key, 3 if key.startswith("seconds") else 1) for key in row
                      if key.startswith(("seconds_", "rss_MB_"))]
        elif name == "nestedbd-100-diagnostics":
            fields = [(key, 3 if key.startswith("clade_") else 1) for key in row
                      if key.startswith(("posterior_ESS_", "likelihood_ESS_", "clade_")) and key != "posterior_ESS_at_least_200"]
        elif name == "ktn152-feasibility":
            fields = [("seconds", 3), ("rss_MiB", 1)]
        elif name == "nestedbd-long-convergence":
            fields = [(key, 2) for key in row if "ESS_" in key]
        else:
            keys = ["mean"]
            if name == "confirmatory-biopsy-gains":
                keys += ["median"]
            if row.get("lower") is not None:
                keys += ["lower", "upper"]
            fields = [(key, int(digits)) for key in keys]
        result.extend(f"{float(row[key]):.{precision}f}" for key, precision in fields)
    return Counter(result)


def audit(supplement, tables):
    text = "\n".join(line for line in Path(supplement).read_text().splitlines()
                     if not line.lstrip().startswith("%"))
    checked = 0
    for name, rows in tables.items():
        marker = "\\label{tab:supp-" + name + "}"
        section = text.split(marker, 1)[1].split("\\end{table*}", 1)[0]
        cells = []
        for tabular in re.findall(r"\\begin\{tabular\}(.*?)\\end\{tabular\}", section, flags=re.S):
            cells.extend(line for line in tabular.splitlines() if "&" in line)
        numbers = Counter(re.findall(r"-?\d+\.\d+", "\n".join(cells).replace("--", ",")))
        expected = printed_values(name, rows)
        if numbers != expected:
            raise ValueError(f"Printed table differs: {name}; printed-only={dict(numbers-expected)}; exported-only={dict(expected-numbers)}")
        if "W/T/L" in "\n".join(cells):
            expected_wtl = Counter(tuple(int(row[key]) for key in ("wins", "ties", "losses"))
                                   for row in rows if row.get("wins") is not None)
            denominators = {sum(row) for row in expected_wtl}
            actual_wtl = Counter(tuple(map(int, match)) for match in
                                 re.findall(r"(?<!\d)(\d+)/(\d+)/(\d+)(?!\d)", "\n".join(cells))
                                 if sum(map(int, match)) in denominators)
            if actual_wtl != expected_wtl:
                raise ValueError(f"Printed W/T/L differs: {name}")
        checked += sum(numbers.values())
    return checked
