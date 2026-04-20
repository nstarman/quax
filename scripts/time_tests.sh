#!/usr/bin/env bash
# time_tests.sh — Run the full quax test suite with per-test timeouts.
#
# Usage:
#   ./scripts/time_tests.sh [TIMEOUT_SECONDS] [extra pytest args...]
#
# Arguments:
#   TIMEOUT_SECONDS  Per-test timeout in seconds. Tests exceeding this are
#                    killed and marked FAILED. Defaults to 60.
#   extra pytest args  Any additional arguments are forwarded to pytest.
#
# Output:
#   test_timings.txt      — clean sorted table: duration | status | test id
#   test_timings_full.log — full pytest output (verbose, for debugging)
#
# Examples:
#   ./scripts/time_tests.sh          # 60s timeout (default)
#   ./scripts/time_tests.sh 10       # 10s timeout
#   ./scripts/time_tests.sh 30 -x    # 30s timeout, stop on first failure

set -uo pipefail

TIMEOUT="${1:-60}"
# Consume the first argument (timeout) if it was provided as a number;
# remaining args are forwarded to pytest.
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    shift
fi

SUMMARY_FILE="test_timings.txt"
FULL_LOG="test_timings_full.log"

echo "================================================================"
echo "  quax test timer"
echo "  per-test timeout : ${TIMEOUT}s"
echo "  summary file     : ${SUMMARY_FILE}"
echo "  full log         : ${FULL_LOG}"
echo "  run started      : $(date)"
echo "================================================================"
echo

# Run pytest via uv, injecting pytest-timeout ephemerally (no lockfile change).
# --timeout         : kill any single test after TIMEOUT seconds
# --timeout-method  : thread (works on macOS; signal requires main thread)
# --durations=0     : collect every test's wall-clock time for post-processing
# --durations-min=0.0 : include even sub-millisecond tests in the report
# -v                : verbose — associates test names with their durations
# "$@"              : forward any remaining CLI arguments
uv run \
    --with pytest-timeout \
    pytest \
    --timeout="${TIMEOUT}" \
    --timeout-method=thread \
    --durations=0 \
    --durations-min=0.0 \
    -v \
    "$@" \
    2>&1 | tee "${FULL_LOG}"

PYTEST_EXIT="${PIPESTATUS[0]}"

echo
echo "================================================================"
echo "  Building duration summary..."
echo "================================================================"

# Extract only the "call" phase durations from the slowest-durations block.
# Format in that block: "  5.68s call     tests/foo.py::test_bar"
# We sort numerically by duration (descending) and also capture FAILED tests.
{
    printf "%-10s  %-8s  %s\n" "DURATION" "STATUS" "TEST"
    printf "%-10s  %-8s  %s\n" "--------" "------" "----"

    # Extract call durations from the pytest durations section
    grep -E "^[[:space:]]*[0-9]+\.[0-9]+s call" "${FULL_LOG}" \
        | sed 's/^[[:space:]]*//' \
        | sort -rn \
        | while read -r duration phase testid; do
            # Look up whether this test PASSED, FAILED, or TIMEOUT in the log
            if grep -q "TIMEOUT ${testid}" "${FULL_LOG}" 2>/dev/null; then
                status="TIMEOUT"
            elif grep -q "FAILED ${testid}" "${FULL_LOG}" 2>/dev/null; then
                status="FAILED"
            else
                status="passed"
            fi
            printf "%-10s  %-8s  %s\n" "${duration}" "${status}" "${testid}"
        done
} > "${SUMMARY_FILE}"

NLINES=$(( $(wc -l < "${SUMMARY_FILE}") - 2 ))
echo
echo "  ${NLINES} tests with call-phase timings written to: ${SUMMARY_FILE}"
echo "  Full pytest output saved to: ${FULL_LOG}"
echo "  run finished : $(date)"
echo "================================================================"
echo
echo "  Top 20 slowest tests:"
head -22 "${SUMMARY_FILE}"

exit "${PYTEST_EXIT}"
