#!/bin/bash

# Usage: ./perf_speedup_plot.sh <binary_name>
BINARY=$1

if [ -z "$BINARY" ]; then
    echo "Usage: $0 <binary_name>"
    exit 1
fi

PERF="/usr/lib/linux-tools/5.15.0-168-generic/perf"

if [ ! -x "$PERF" ]; then
    echo "ERROR: perf not found at $PERF"
    exit 1
fi

OUTPUT_FILE="speedup_results.csv"

echo "Threads,Time_Seconds,Speedup,Efficiency(%)" > "$OUTPUT_FILE"

THREAD_COUNTS=(1 2 4 8 16 24)

echo "Starting PERF-based speedup analysis for $BINARY..."
echo "-----------------------------------------------"

T1=""

for T in "${THREAD_COUNTS[@]}"; do

    echo "Running with $T threads..."
    export OMP_NUM_THREADS=$T

    $PERF stat -o perf_tmp.txt ./"$BINARY" > /dev/null

    TIME=$(grep "seconds time elapsed" perf_tmp.txt | awk '{print $1}')

    if [ -z "$TIME" ]; then
        echo "ERROR extracting time"
        continue
    fi

    if [ "$T" -eq 1 ]; then
        T1=$TIME
        SPEEDUP=1
    else
        SPEEDUP=$(awk -v t1="$T1" -v tn="$TIME" 'BEGIN { printf "%.6f", t1/tn }')
    fi

    EFFICIENCY=$(awk -v s="$SPEEDUP" -v t="$T" 'BEGIN { printf "%.2f", (s/t)*100 }')

    echo "$T,$TIME,$SPEEDUP,$EFFICIENCY" >> "$OUTPUT_FILE"

done

rm -f perf_tmp.txt

echo "-----------------------------------------------"
echo "Data saved to $OUTPUT_FILE"

# ---------------------------
# AUTO-PLOT USING PYTHON
# ---------------------------

python3 << EOF

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("$OUTPUT_FILE")

threads = df["Threads"]
speedup = df["Speedup"]
efficiency = df["Efficiency(%)"]
time = df["Time_Seconds"]

# Speedup graph
plt.figure()
plt.plot(threads, speedup, marker="o", label="Measured Speedup")
#plt.plot(threads, threads, linestyle="--", label="Ideal Speedup")
plt.xlabel("Threads")
plt.ylabel("Speedup")
plt.title("Speedup vs Threads")
plt.xticks(range(0, int(max(threads)) + 2, 2))
plt.grid(True)
plt.legend()
plt.savefig("speedup.png")

# Execution time graph
plt.figure()
plt.plot(threads, time, marker="o", color="red")
plt.xlabel("Threads")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time vs Threads")
plt.xticks(range(0, int(max(threads)) + 2, 2))
plt.grid(True)
plt.savefig("execution_time.png")

# Efficiency graph
plt.figure()
plt.plot(threads, efficiency, marker="o", color="green")
plt.xlabel("Threads")
plt.ylabel("Efficiency (%)")
plt.title("Efficiency vs Threads")
plt.xticks(range(0, int(max(threads)) + 2, 2))
plt.grid(True)
plt.savefig("efficiency.png")

print("All graphs generated")

EOF

echo "-----------------------------------------------"
echo "Graphs generated:"
echo "speedup.png"
echo "execution_time.png"
echo "efficiency.png"

# ---------------------------
# PRINT TABLE
# ---------------------------

echo ""
echo "-----------------------------------------------"
printf '%s\n' 'Threads | Execution Time | Speedup | Efficiency (%)'
echo "-----------------------------------------------"

awk -F',' '
NR==1 {next}
{
printf "%-7d | %-14.6f | %-7.2f | %-10.2f%%\n", $1,$2,$3,$4
}
' "$OUTPUT_FILE"

echo "-----------------------------------------------"
