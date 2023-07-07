#!/bin/bash

TraceTag='trace'

if [ $# -ne 4 ]
  then
    echo "Usage train.sh <trace_1> <trace_2> <inflection_percentile> <granularity>"
    # eg : ./train.sh testTraces/hacktest.trace testTraces/hacktest.trace testTraces/hacktest.trace 85 4
    exit
fi

echo $1, $2, $3, $4

sudo ../io_replayer/replayer baseline mlData/TrainTraceOutput 2 /dev/nvme0n1-/dev/sda $1 $2

# pip3 install numpy
# pip3 install --upgrade pip
# pip3 install tensorflow
# pip3 install keras
# pip3 install pandas
# pip3 install scikit-learn

mkdir -p mlData
for i in 0 1
do
   python3 traceParser.py direct 3 4 \
   mlData/TrainTraceOutput_baseline.data mlData/temp1 \
   mlData/"mldrive${i}" "$i" $4
done

# For granularity = 1
for i in 0 1
do
   python3 pred1_+1.py \
   mlData/"mldrive${i}_gran_1.csv" $3 1 > mlData/"mldrive${i}.txt.gran_1_results".txt
done

# For specified granularity
if [ $4 -gt 1 ]
   then
      for i in 0 1
      do
         python3 pred1_+1.py \
         mlData/"mldrive${i}_gran_2.csv" $3 $4 > mlData/"mldrive${i}.txt.gran_1_results".txt
      done
fi

cd mlData
mkdir -p drive0weights
mkdir -p drive1weights
cp mldrive0_gran* drive0weights
cp mldrive1_gran* drive1weights

cd ..
mkdir -p weights_header_2ssds

# For granularity = 1
python3 mlHeaderGen+1.py Trace nvme0n1 mlData/drive0weights weights_header_2ssds 1
python3 mlHeaderGen+1.py Trace sda mlData/drive0weights weights_header_2ssds 1

# For granularity = 2
if [ $4 -gt 1 ]
   then
      python3 mlHeaderGen+1.py Trace nvme0n1 mlData/drive1weights weights_header_2ssds $4
fi
if [ $4 -gt 1 ]
   then
      python3 mlHeaderGen+1.py Trace sda mlData/drive1weights weights_header_2ssds $4
fi

cd ../kernel_hook/weights_header/mix
cp ../../../training/weights_header_2ssds/* .