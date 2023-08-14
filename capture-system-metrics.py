import subprocess
import sys
import time

import psutil


# power1, memory1, gpu utilization1, power2, memory2, gpu utilization2, power3, memory3, gpu utilization3
def capture_gpu_metrics():
    result = subprocess.run(["nvidia-smi | grep 300W"], shell=True, capture_output=True)

    metrics = []
    lines = result.stdout.decode("UTF-8").replace("|", "").replace("/", "").split("\n")
    for line in lines[:3]:
        val = line.split()
        metrics.append("{},{},{}".format(
            val[3].replace("W", ""),
            val[5].replace("MiB", ""),
            val[7].replace("%", ""))
        )
    return ",".join(metrics)


def get_process_ids():
    result = subprocess.run(["nvidia-smi | grep python3"], shell=True, capture_output=True)
    pids = []
    lines = result.stdout.decode("UTF-8").replace("|", "").split("\n")
    for line in lines[:-1]:
        val = line.split()
        pids.append(int(val[3]))
    return pids


# cpu util %, memory util %,
def capture_cpu_metrics():
    pids = get_process_ids()
    metrics = []
    for pid in pids:
        proc = psutil.Process(pid)
        cpu_pct = proc.cpu_percent(interval=0.1)
        mem_pct = proc.memory_percent()
        mem_usage = proc.memory_info().rss
        io_counters = proc.io_counters()
        read_bytes = io_counters.read_bytes
        write_bytes = io_counters.write_bytes
        metrics.append("{},{},{},{},{}".format(cpu_pct, mem_pct, mem_usage, read_bytes, write_bytes))
    return ",".join(metrics)


while True:
    filename = sys.argv[1]
    sleep_duration = sys.argv[2]
    try:
        gpu_metrics = capture_gpu_metrics()
        cpu_metrics = capture_cpu_metrics()
        with open(filename, "a") as f:
            f.write("{},{}\n".format(gpu_metrics, cpu_metrics))
        time.sleep(float(sleep_duration))
    except Exception as e:
        with open(filename, "a") as f:
            f.write(e)
        break
