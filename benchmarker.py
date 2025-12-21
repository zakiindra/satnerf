import pathlib
import importlib
import argparse
import time
import threading
import subprocess
import os
import re
from datetime import datetime
from timeit import default_timer as timer

class CPU(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.result = None
        self.event = threading.Event()
        self._list = []

    def run(self):
        try:
            while not self.event.is_set():
                output = subprocess.check_output([
                    'pidstat', '-p', str(os.getpid()), '1', '1'])
                cpu_ = float(output.splitlines()[-2].split()[-3])
                self._list.append(cpu_)

            self.event.clear()
            res = sum(self._list) / len(self._list)
            self.result = res, self._list, output
        except:
            self.result = 0, self._list, output

    def stop(self):
        self.event.set()


class Memory(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.result = None
        self.event = threading.Event()
        self._list = []

    def run(self):
        try:
            while not self.event.is_set():
                output = subprocess.check_output([
                    'pidstat', '-p', str(os.getpid()), '1', '1', '-r'])
                mem_ = float(output.splitlines()[-2].split()[-3])
                self._list.append(mem_)

            self.event.clear()
            res = sum(self._list) / len(self._list)
            self.result = res, self._list, output
        except:
            self.result = 0, self._list, output

    def stop(self):
        self.event.set()


def jstat_start():
    subprocess.check_output(
        f'tegrastats --interval 1000 --start --logfile test.txt',
        shell=True)


def jstat_stop():
    subprocess.check_output(f'tegrastats --stop', shell=True)

    cpu_list = []
    gpu_list = []
    ram_used_list = []
    swap_list = []
    total_vdd_list = []

    with open("test.txt", "r") as f:
        for line in f:
            pattern = re.compile(
                r"RAM\s+(\d+)/\d+MB.*?"
                r"SWAP\s+(\d+)/\d+MB.*?"
                r"CPU\s*\[([^\]]+)\].*?"
                r"GR3D_FREQ\s+(\d+)%.*?"
                r"VDD_GPU_SOC\s+(\d+)(?:mW)?.*?"
                r"VDD_CPU_CV\s+(\d+)(?:mW)?.*?"
                r"VIN_SYS_5V0\s+(\d+)(?:mW)?",
                re.DOTALL
            )

            match = pattern.search(line)
            if match:
                ram, swap, cpu_raw, gr3d, vdd_1, vdd_2, vdd_3 = match.groups()
                ram, swap, gr3d, vdd_1, vdd_2, vdd_3 = float(ram), float(swap), float(gr3d.rstrip('%')), float(vdd_1), float(vdd_2), float(vdd_3)

                # extract CPU percentages as integers
                cpu_values = [float(x) for x in re.findall(r"(\d+)(?=%@)", cpu_raw)]
                cpu_total = sum(cpu_values)

                total_vdd = vdd_1 + vdd_2 + vdd_3

                cpu_list.append(cpu_total)
                gpu_list.append(gr3d)
                ram_used_list.append(ram)
                swap_list.append(swap)
                total_vdd_list.append(total_vdd)

    avg_cpu = sum(cpu_list) / len(cpu_list) if cpu_list else 0.0
    avg_gpu = sum(gpu_list) / len(gpu_list) if gpu_list else 0.0
    avg_ram_used = sum(ram_used_list) / len(ram_used_list) if ram_used_list else 0.0
    avg_swap = sum(swap_list) / len(swap_list) if swap_list else 0.0
    avg_vdd = sum(total_vdd_list) / len(total_vdd_list) if total_vdd_list else 0.0

    os.remove("test.txt")

    return avg_cpu, avg_gpu, avg_ram_used, avg_swap, avg_vdd

