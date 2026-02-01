#!/bin/bash

# Usage: ./perf_speedup_plot.sh <binary_name>
BINARY=$1

if [ -z "$BINARY" ]; then
    echo "Usage: $0 <binary_name>"
    exit 1
fi

# Path to perf (your alias path)
PERF="/usr/lib/linux-tools/5.15.0-168-generic/perf"

if [ ! -x "$PERF" ]; then
    echo "ERROR: perf not found at $PERF"
    exit 1
fi

OUTPUT_FILE="speedup_results.csv"
PLOT="speedup_graph.png"

echo "Threads,Time_Seconds,Speedup" > "$OUTPUT_FILE"

THREAD_COUNTS=(1 2 4 8 16 24)

echo "Starting PERF-based speedup analysis for $BINARY..."
echo "-----------------------------------------------"

T1=""

for T in "${THREAD_COUNTS[@]}"; do
    echo "Running with $T threads..."
    export OMP_NUM_THREADS=$T

    # Run perf and capture output
    $PERF stat -o perf_tmp.txt ./"$BINARY" > /dev/null

    # Extract elapsed time
    TIME=$(grep "seconds time elapsed" perf_tmp.txt | awk '{print $1}')

    if [ -z "$TIME" ]; then
        echo "ERROR: Could not extract time"
        cat perf_tmp.txt
        continue
    fi

    echo "Time = $TIME seconds"

    if [ "$T" -eq 1 ]; then
        T1=$TIME
        SPEEDUP=1
    else
        SPEEDUP=$(awk -v t1="$T1" -v tn="$TIME" 'BEGIN { printf "%.6f", t1/tn }')
    fi

    echo "$T,$TIME,$SPEEDUP" >> "$OUTPUT_FILE"
done

rm -f perf_tmp.txt

echo "-----------------------------------------------"
echo "Data saved to $OUTPUT_FILE"

# ---------------------------
# AUTO-PLOT USING PYTHON
# ---------------------------
python3 - << EOF
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("$OUTPUT_FILE")

threads = df["Threads"]
speedup = df["Speedup"]

plt.figure()

# Measured speedup
plt.plot(threads, speedup, marker="o", label="Measured Speedup")

# Ideal 45-degree line (y = x)
plt.plot(threads, threads, linestyle="--", label="Ideal Speedup (y = x)")

plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.title("Speedup vs Number of Threads")
plt.grid(True)
plt.legend()

plt.savefig("$PLOT")
print("Graph saved as $PLOT")
EOF

echo "-----------------------------------------------"
echo "ALL DONE!"
echo "CSV:   $OUTPUT_FILE"
echo "Graph: $PLOT"

# ---------------------------
# PRINT FORMATTED TABLE
# ---------------------------
echo ""
echo "-----------------------------------------------"
printf '%s\n' 'Threads N | Execution Time (s) | Speedup (S) | Efficiency (E=S/N)'
echo "-----------------------------------------------"

awk -F',' '
NR==1 { next }
{
    N=$1
    T=$2
    S=$3
    E=(S/N)*100

    if (N==1)
        printf "%-16s | %-18.6f | %-10.2fx | %-6.1f%%\n", "1 Sequential", T, S, E
    else
        printf "%-16d | %-18.6f | %-10.2fx | %-6.1f%%\n", N, T, S, E
}
' speedup_results.csv

echo "-----------------------------------------------"
