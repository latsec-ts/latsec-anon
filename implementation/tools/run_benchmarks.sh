#!/bin/sh
# Run the measurements of Section 8 and Appendix P of the full version: the six parameter sets of
# Table 4 at three signing-set sizes each, one after the other, on one core.
#
#     tools/run_benchmarks.sh [PREFIX [CPU]]
#
# Run from the directory of the implementation, after `cargo build --release` in
# latsec-rs.  PREFIX names the records, results/PREFIX-author-n<n>-k<kappa>-act<size>
# with the extensions .json and .log (default: measure-<date>).  About 15 minutes.
#
# CPU is the core the runs are pinned to, as `taskset -c` takes it.  Without it the
# script chooses among the cores this process may run on: one with the highest
# maximum frequency (machines with cores of different speeds), and not core 0 if
# there is a choice (it tends to serve more interrupts).  Choose an otherwise idle
# core yourself where you can.  Without taskset the runs are not pinned.
#
# The seeds are fixed, so the signer sets, the encoded sizes and the norms of a run
# are the same on every machine; the times are not.  The records in the paper were
# written with PREFIX = 20261116-measure.  tools/bench_tables.py --prefix PREFIX
# turns the records into the tables of the paper.
set -e
prefix=${1:-measure-$(date +%Y%m%d)}
latsec=latsec-rs/target/release/latsec
[ -x "$latsec" ] || { echo "$latsec not found: run 'cargo build --release' in latsec-rs first" >&2; exit 1; }

# The cores this process may run on, one per line ("0-3,8" -> 0 1 2 3 8).
allowed_cpus() {
  taskset -cp $$ 2>/dev/null | sed 's/.*: *//' | tr ',' '\n' | while IFS=- read -r first last; do
    seq "$first" "${last:-$first}"
  done
}

# Among the allowed cores, one with the highest maximum frequency, avoiding core 0
# when another core is as fast.  Cores without a frequency entry count as equal.
choose_cpu() {
  best=; best_freq=-1
  for c in $(allowed_cpus); do
    freq=$(cat "/sys/devices/system/cpu/cpu$c/cpufreq/cpuinfo_max_freq" 2>/dev/null || echo 0)
    if [ "$freq" -gt "$best_freq" ] || { [ "$freq" -eq "$best_freq" ] && [ "$best" = 0 ]; }; then
      best=$c; best_freq=$freq
    fi
  done
  echo "$best"
}

if ! command -v taskset >/dev/null 2>&1; then
  pin=""
  echo "taskset not found: the runs are not pinned to a core" >&2
else
  cpu=${2:-$(choose_cpu)}
  if [ -z "$cpu" ] || ! taskset -c "$cpu" true 2>/dev/null; then
    echo "cannot pin to core '$cpu': give a core this process may run on (allowed: $(allowed_cpus | tr '\n' ' '))" >&2
    exit 1
  fi
  pin="taskset -c $cpu "
  echo "pinning the runs to core $cpu"
fi

for n in 40 60; do
  if [ "$n" = 40 ]; then sizes="26 33 40"; else sizes="46 53 60"; fi
  for kappa in 128 192 256; do
    sessions=100            # at the signing threshold; 30 for the larger sizes
    for size in $sizes; do
      base=results/$prefix-author-n$n-k$kappa-act$size
      cmd="$pin$latsec measure --profile profiles/author-n$n-k$kappa-nov2026.json --size $size --sessions $sessions --warmup 5 --keygen-reps 11 --seed 20261116-n$n-k$kappa-act$size --json $base.json"
      { echo "# $cmd"; $cmd; } > "$base.log" 2>&1
      echo "wrote $base.json"
      sessions=30
    done
  done
done
