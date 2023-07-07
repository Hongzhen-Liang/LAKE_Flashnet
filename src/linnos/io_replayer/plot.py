import numpy as np
import matplotlib.pyplot as plt
def plot_raw_vs_best(figure_path, y_raw, y_cpu, y_gpu2, y_gpu1, y_gpu3, y_gpu4, y_gpu6,  y_gpu8, extra_info=""):
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

    N=len(y_gpu2)
    data = y_gpu2
    # sort the data in ascending order
    x_3 = np.sort(data)
    # get the cdf values of y
    y_3 = np.arange(N) / float(N)

    N=len(y_gpu1)
    data = y_gpu1
    # sort the data in ascending order
    x_4 = np.sort(data)
    # get the cdf values of y
    y_4 = np.arange(N) / float(N)

    N=len(y_gpu3)
    data = y_gpu3
    # sort the data in ascending order
    x_5 = np.sort(data)
    # get the cdf values of y
    y_5 = np.arange(N) / float(N)

    N=len(y_gpu4)
    data = y_gpu4
    # sort the data in ascending order
    x_6 = np.sort(data)
    # get the cdf values of y
    y_6 = np.arange(N) / float(N)

    N=len(y_gpu6)
    data = y_gpu6
    # sort the data in ascending order
    x_7 = np.sort(data)
    # get the cdf values of y
    y_7 = np.arange(N) / float(N)

    N=len(y_gpu8)
    data = y_gpu8
    # sort the data in ascending order
    x_8 = np.sort(data)
    # get the cdf values of y
    y_8 = np.arange(N) / float(N)

    # plotting
    plt.figure(figsize=(7,3))
    plt.xlabel('Latency (us)')
    plt.ylabel('CDF')
    plt.title('CDF of Latency (' + worload + ' IOs) \n' + extra_info)
    p70_lat = np.percentile(y_raw, 70)
    plt.xlim(0, max(10000, p70_lat * 4)) # Hopefully the x axis limit can catch the tail
    plt.ylim(0.9, 1) 
    plt.plot(x_1, y_1, label = x1_label, color="green")
    plt.plot(x_2, y_2, label = x2_label, color="red")
    plt.plot(x_3, y_3, label = x3_label, color="blue")
    plt.plot(x_4, y_4, label = x4_label, color="orange")
    # plt.plot(x_5, y_5, label = x5_label, color="blue")
    # plt.plot(x_6, y_6, label = x6_label, color="red")
    # plt.plot(x_8, y_8, label = x8_label, color="black")
    plt.legend(loc="lower right")
    plt.legend(fontsize=7)
    plt.savefig(figure_path, bbox_inches='tight')
    print("===== output figure : " + figure_path)

baseline = open("1ssd_baseline.data")
cpu = open("1ssd_cpu.data")

# gran = 1
gpu_1 = open("1ssd_gpu_gran1.data")
gran1 = 1

# gran = 2
gpu_2 = open("1ssd_gpu_gran2.data")
gran2 = 2

gpu_3 = open("1ssd_gpu_gran3.data")
gran3 = 3

gpu_4 = open("1ssd_gpu_gran4.data")
gran4 = 4

# # gran = 6
# gpu_6 = open("1ssd_gpu_gran6.data")
# gran6 = 6

# # gran = 8
# gpu_8 = open("1ssd_gpu_gran8.data")
# gran8 = 8

x1_label = "Linnos - disable"
x2_label = "Linnos - enable - CPU"
x3_label = "Linnos - enable - GPU - gran = " + str(gran2)
x4_label = "Linnos - enable - GPU - gran = " + str(gran1)
x5_label = "Linnos - enable - GPU - gran = " + str(gran3)
x6_label = "Linnos - enable - GPU - gran = " + str(gran4)
# x8_label = "Linnos - enable - GPU - gran = " + str(gran8)

worload = "read"

lineBase = baseline.readline()
lineCPU = cpu.readline()
lineGPU1 = gpu_1.readline() 
lineGPU2 = gpu_2.readline() 
lineGPU3 = gpu_3.readline() 
lineGPU4 = gpu_4.readline() 
# lineGPU6 = gpu_6.readline() 
# lineGPU8 = gpu_8.readline() 

LB = []
LCPU = []
LGPU1 = []
LGPU2 = []
LGPU3 = []
LGPU4 = []
LGPU6 = []
LGPU8 = []
 
while (lineBase and lineCPU and lineGPU1 and lineGPU2 and lineGPU3 and lineGPU4):
    if worload == "read":
        if int(lineBase.split(",")[2]) == 1:
            LB.append(int(lineBase.split(",")[1]))
        if int(lineCPU.split(",")[2]) == 1:
            LCPU.append(int(lineCPU.split(",")[1]))
        if int(lineGPU2.split(",")[2]) == 1:
            LGPU2.append(int(lineGPU2.split(",")[1]))
        if int(lineGPU1.split(",")[2]) == 1:
            LGPU1.append(int(lineGPU1.split(",")[1]))
        if int(lineGPU3.split(",")[2]) == 1:
            LGPU3.append(int(lineGPU3.split(",")[1]))
        if int(lineGPU4.split(",")[2]) == 1:
            LGPU4.append(int(lineGPU4.split(",")[1]))
        # if int(lineGPU6.split(",")[2]) == 1:
        #     LGPU6.append(int(lineGPU6.split(",")[1]))
        # if int(lineGPU8.split(",")[2]) == 1:
        #     LGPU8.append(int(lineGPU8.split(",")[1]))
    elif worload == "write":
        if int(lineBase.split(",")[2]) == 0:
            LB.append(int(lineBase.split(",")[1]))
        if int(lineCPU.split(",")[2]) == 0:
            LCPU.append(int(lineCPU.split(",")[1]))
        if int(lineGPU2.split(",")[2]) == 0:
            LGPU2.append(int(lineGPU2.split(",")[1]))
        if int(lineGPU1.split(",")[2]) == 0:
            LGPU1.append(int(lineGPU1.split(",")[1]))
        if int(lineGPU3.split(",")[2]) == 0:
            LGPU3.append(int(lineGPU3.split(",")[1]))
        if int(lineGPU4.split(",")[2]) == 0:
            LGPU4.append(int(lineGPU4.split(",")[1]))
        # if int(lineGPU6.split(",")[2]) == 0:
        #     LGPU6.append(int(lineGPU6.split(",")[1]))
        # if int(lineGPU8.split(",")[2]) == 0:
        #     LGPU8.append(int(lineGPU8.split(",")[1]))
    elif worload == "all":
        LB.append(int(lineBase.split(",")[1]))
        LCPU.append(int(lineCPU.split(",")[1]))
        LGPU2.append(int(lineGPU2.split(",")[1]))
        LGPU1.append(int(lineGPU1.split(",")[1]))
        LGPU3.append(int(lineGPU3.split(",")[1]))
        LGPU4.append(int(lineGPU4.split(",")[1]))
        # LGPU6.append(int(lineGPU6.split(",")[1]))
        # LGPU8.append(int(lineGPU8.split(",")[1]))
    lineBase = baseline.readline()
    lineCPU = cpu.readline()
    lineGPU2 = gpu_2.readline()
    lineGPU1 = gpu_1.readline()
    lineGPU3 = gpu_3.readline()
    lineGPU4 = gpu_4.readline()
    # lineGPU6 = gpu_6.readline()
    # lineGPU8 = gpu_8.readline()

plot_raw_vs_best("Baseline_CPU_GPU_" + worload + ".png",LB,LCPU,LGPU2,LGPU1,LGPU3,LGPU4,LGPU6,LGPU8,"[Lake power LinnOS]")
