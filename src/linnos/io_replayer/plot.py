import numpy as np
import matplotlib.pyplot as plt
def plot_raw_vs_best(figure_path, y_raw, y_best, extra_info=""):
    # Draw CDF
    N=len(y_best)
    data = y_best
    # sort the data in ascending order
    x_1 = np.sort(data)
    # get the cdf values of y
    y_1 = np.arange(N) / float(N)

    N=len(y_raw)
    data = y_raw
    # sort the data in ascending order
    x_2 = np.sort(data)
    # get the cdf values of y
    y_2 = np.arange(N) / float(N)

    # plotting
    plt.figure(figsize=(7,3))
    plt.xlabel('Latency (us)')
    plt.ylabel('CDF')
    plt.title('CDF of Latency (Read-only IOs) \n' + extra_info)
    p70_lat = np.percentile(y_raw, 70)
    plt.xlim(0, max(p70_lat * 3, 1000)) # Hopefully the x axis limit can catch the tail
    plt.ylim(0, 1) 
    plt.plot(x_2, y_2, label = "Raw latency", color="red")
    plt.plot(x_1, y_1, label = "FlashNet-best-case", color="green")
    plt.legend(loc="lower right")
    plt.savefig(figure_path, bbox_inches='tight')
    print("===== output figure : " + figure_path)

baseline = open("2ssds_baseline.data")
failover = open("2ssds_failover.data")

lineBase = baseline.readline()
lineFailover = failover.readline()

LB = []
LF = []
 
while lineBase:
    LB.append(int(lineBase.split(",")[1]))
    LF.append(int(lineFailover.split(",")[1]))
    lineBase = baseline.readline()
    lineFailover = failover.readline()

plot_raw_vs_best("Baseline_Failover.png",LB,LF,"[Lake power LinnOS]")
