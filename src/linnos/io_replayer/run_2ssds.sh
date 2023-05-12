#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "Need type, baseline or failover and 3 trace files"
fi

#../../../benchmarks/cpu_utilization/kutil_kill > backingfile &
#PID=$!
# ./run_2ssds.sh baseline ../trace_tools/hongzhen/alibaba.cut.per_10k.most_size_thpt_iops_rand.719.trace ../trace_tools/hongzhen/msr.cut.per_10k.most_thpt_rand_iops.1006.trace
sudo ./replayer $1 2ssds 2 /dev/sda-/dev/sdc $2 $3

#sudo ./replayer $1 3ssds 2 /dev/nvme0n1-/dev/nvme2n1 $2 $3 $4
#kill -SIGINT  $PID
#out=$(cat backingfile)
#echo "Average kernel cpu%: $out"

