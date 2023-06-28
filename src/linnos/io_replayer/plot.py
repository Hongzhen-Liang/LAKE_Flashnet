import numpy as np
import matplotlib.pyplot as plt
def plot_raw_vs_best(figure_path, y_raw, y_cpu, y_gpu, extra_info=""):
    # Draw CDF
    N=len(y_raw)
    data = y_raw
    # sort the data in ascending order
    x_1 = np.sort(data)
    # get the cdf values of y
    y_1 = np.arange(N) / float(N)

    N=len(y_cpu)
    data = y_cpu
    # sort the data in ascending order
    x_2 = np.sort(data)
    # get the cdf values of y
    y_2 = np.arange(N) / float(N)

    N=len(y_gpu)
    data = y_gpu
    # sort the data in ascending order
    x_3 = np.sort(data)
    # get the cdf values of y
    y_3 = np.arange(N) / float(N)

    # plotting
    plt.figure(figsize=(7,3))
    plt.xlabel('Latency (us)')
    plt.ylabel('CDF')
    plt.title('CDF of Latency (Read-only IOs) \n' + extra_info)
    p70_lat = np.percentile(y_raw, 70)
    plt.xlim(0, max(p70_lat * 3, 100)) # Hopefully the x axis limit can catch the tail
    plt.ylim(0, 1) 
    plt.plot(x_1, y_1, label = x1_label, color="green")
    plt.plot(x_2, y_2, label = x2_label, color="red")
    plt.plot(x_3, y_3, label = x3_label, color="blue")
    plt.legend(loc="lower right")
    plt.savefig(figure_path, bbox_inches='tight')
    print("===== output figure : " + figure_path)

baseline = open("1ssd_baseline.data")
cpu = open("1ssd_cpu.data")
gpu = open("1ssd_gpu.data")

x1_label = "Linnos - disable"
x2_label = "Linnos - enable - CPU"
x3_label = "Linnos - enable - GPU"

lineBase = baseline.readline()
lineCPU = cpu.readline()
lineGPU = gpu.readline() 

LB = []
LCPU = []
LGPU = []
 
while (lineBase and lineCPU and lineGPU):
    LB.append(int(lineBase.split(",")[1]))
    LCPU.append(int(lineCPU.split(",")[1]))
    LGPU.append(int(lineGPU.split(",")[1]))
    lineBase = baseline.readline()
    lineCPU = cpu.readline()
    lineGPU = gpu.readline()

plot_raw_vs_best("Baseline_CPU_GPU.png",LB,LCPU,LGPU,"[Lake power LinnOS]")
