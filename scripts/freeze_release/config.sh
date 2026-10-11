# Shared owner configuration. Each step reads this file independently.
# Environment overrides are supported for testing or another checkout.
# This is not a scientific runtime manifest.
CTBF_ROOT="${CTBF_ROOT:-$(cd "$FREEZE_SCRIPT_DIR/../.." && pwd -P)}"
CTBFBIO_ROOT="${CTBFBIO_ROOT:-/Users/voronwe/Work/ctbfbio}"
CTBF_RELEASE_WORK="${CTBF_RELEASE_WORK:-/Users/voronwe/Work/ctbf_release_1.2.0}"
CTBF_RELEASE_TAG="${CTBF_RELEASE_TAG:-v1.2.0}"
CTBFBIO_RELEASE_TAG="${CTBFBIO_RELEASE_TAG:-v1.0.0}"
CTBF_GITHUB="${CTBF_GITHUB:-j-paszek/ctbf}"
CTBFBIO_GITHUB="${CTBFBIO_GITHUB:-j-paszek/ctbfbio}"

# This interpreter passed the original audit. Change this ONE setting if
# using a different environment. Do not silently fall back to another Python.
CTBF_PYTHON="${CTBF_PYTHON:-/Users/voronwe/Work/PyCharmProjects/sel/.venv/bin/python}"
