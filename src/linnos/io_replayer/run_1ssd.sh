#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "Need type, baseline or failover and 3 trace files"
fi

#../../../benchmarks/cpu_utilization/kutil_kill > backingfile &
#PID=$!

sudo ./replayer $1 1ssd 1 /dev/sdb $2

#sudo ./replayer $1 3ssds 2 /dev/nvme0n1-/dev/nvme2n1 $2 $3 $4
#kill -SIGINT  $PID
#out=$(cat backingfile)
#echo "Average kernel cpu%: $out"

