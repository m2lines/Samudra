#!/bin/bash
# Per-minute kernel memory stats on this node, to line data-load stalls up against reclaim and compaction.
# srun --jobid <job> --overlap -N <n> --ntasks-per-node=1 scripts/_bench/mem_sampler.sh <out_dir> [seconds]
# Columns: time, then per-interval deltas of the vmstat counters, then MemFree/Cached GB and %sys/%iowait CPU.

out="${1:?out dir}/$(hostname -s).tsv"
every="${2:-60}"
keys="allocstall_movable pgscan_direct compact_stall compact_fail thp_fault_alloc thp_fault_fallback"
mkdir -p "$(dirname "${out}")"
[[ -s "${out}" ]] || echo -e "time\t${keys// /\\t}\tfree_gb\tcached_gb\tsys_pct\tiowait_pct" > "${out}"

counters() { awk -v keys="${keys}" 'BEGIN { n = split(keys, k, " ") } { v[$1] = $2 } END { for (i = 1; i <= n; i++) printf "%s ", v[k[i]] }' /proc/vmstat; }
cpu() { awk '/^cpu / { print $2 + $3 + $4 + $5 + $6 + $7 + $8, $4, $6 }' /proc/stat; }

read -r -a before <<< "$(counters)"
read -r total0 sys0 wait0 <<< "$(cpu)"
while sleep "${every}"; do
  read -r -a after <<< "$(counters)"
  read -r total1 sys1 wait1 <<< "$(cpu)"
  line="$(date +%FT%T)"
  for i in "${!after[@]}"; do line+=$'\t'$(( after[i] - before[i] )); done
  mem=$(awk '/^MemFree:/ { f = $2 } /^Cached:/ { c = $2 } END { printf "%.0f\t%.0f", f / 2^20, c / 2^20 }' /proc/meminfo)
  dt=$(( total1 - total0 ))
  echo -e "${line}\t${mem}\t$(( 100 * (sys1 - sys0) / dt ))\t$(( 100 * (wait1 - wait0) / dt ))" >> "${out}"
  before=("${after[@]}"); total0=${total1}; sys0=${sys1}; wait0=${wait1}
done
