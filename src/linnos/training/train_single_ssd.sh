#!/bin/bash

TraceTag='trace'

if [ $# -ne 3 ]
  then
    echo "Usage train.sh <trace_1> <inflection_percentile> <granularity>"
    # eg : ./train.sh testTraces/hacktest.trace testTraces/hacktest.trace testTraces/hacktest.trace 85 3
    exit
fi

echo $1, $2, $3

sudo ../io_replayer/replayer baseline mlData/TrainTraceOutput 1 /dev/sda $1 

pip3 install numpy
pip3 install --upgrade pip
pip3 install tensorflow
pip3 install keras
pip3 install pandas
pip3 install scikit-learn

mkdir -p mlData
for i in 0 
do
   python3 traceParser.py direct 3 4 \
   mlData/TrainTraceOutput_baseline.data mlData/temp1 \
   mlData/"mldrive${i}.csv" "$i" $3
done

for i in 0 
do
   python3 pred1.py \
   mlData/"mldrive${i}.csv" $2 $3 > mlData/"mldrive${i}results".txt
done

cd mlData
mkdir -p drive0weights
cp mldrive0.csv.* drive0weights

cd ..
mkdir -p weights_header_1ssd
python3 mlHeaderGen.py Trace sda mlData/drive0weights weights_header_1ssd