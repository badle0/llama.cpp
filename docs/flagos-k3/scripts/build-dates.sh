#!/bin/sh
# The K3's clock can fall back to an old date (seen 2026-11-17: November builds, then a clock in April). Files dated
# after the clock's "now" break the build: ninja skips changed sources, or CMake re-runs until ninja gives up.
# Usage, from the repo root: build-dates.sh check [build]        says whether any file is dated in the future
#                            build-dates.sh fix <base> [build]   re-dates in order: every tracked file, the build
#                                                                tree, then the files changed since commit <base>
#                                                                (the commit the build tree was last built at)
# It changes file dates in the repo and the build tree only, never the clock. Not needed once the clock is right.
set -u
mode=${1:-check}
if [ "$mode" = fix ]; then base=${2:?fix needs the commit the build tree was built at}; build=${3:-build-flagos}; else build=${2:-build-flagos}; fi
g() { git -c safe.directory="$PWD" "$@"; }
ref=$(mktemp) || exit 1
n_build=$(find "$build" -type f -newer "$ref" | wc -l)
n_src=$(g ls-files -z | xargs -0 sh -c 'for f; do [ "$f" -nt "$0" ] && echo "$f"; done' "$ref" | wc -l)
rm -f "$ref"
echo "clock: $(date -u +%Y-%m-%dT%H:%MZ); files dated after it: build tree $n_build, tracked $n_src"
if [ "$mode" != fix ]; then
    [ "$n_build" -eq 0 ] && [ "$n_src" -eq 0 ] && echo "dates fine: build as usual" && exit 0
    echo "dates ahead of the clock: run '$0 fix <base>' before building"
    exit 1
fi
changed=$(g diff --name-only "$base" HEAD | grep -v '^docs/') || true
g ls-files -z | xargs -0 touch && sleep 2 && find "$build" -type f -exec touch {} + && sleep 2 || exit 1
for f in $changed; do [ -f "$f" ] && touch "$f"; done
echo "re-dated; marked as changed since $base: $(echo "$changed" | grep -c .) file(s)"
