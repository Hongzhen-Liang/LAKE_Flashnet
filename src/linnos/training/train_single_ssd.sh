#!/bin/bash

TraceTag='trace'

if [ $# -ne 3 ]
  then
    echo "Usage train.sh <trace_1> <inflection_percentile> <granularity>"
    # eg : ./train.sh testTraces/hacktest.trace testTraces/hacktest.trace testTraces/hacktest.trace 85 3
    exit
fi

echo $1, $2, $3

# sudo ../io_replayer/replayer baseline mlData/TrainTraceOutput 1 /dev/sdb1 $1 

# pip3 install numpy
# pip3 install --upgrade pip
# pip3 install tensorflow
# pip3 install keras
# pip3 install pandas
# pip3 install scikit-learn

# mkdir -p mlData
# for i in 0 
# do
#    python3 traceParser.py direct 3 4 \
#    mlData/TrainTraceOutput_baseline.data mlData/temp1 \
#    mlData/"mldrive${i}" "$i" $3
# done

# # For granularity == 1
# for i in 0 
# do
#    python3 pred1.py \
#    mlData/"mldrive${i}_gran_1.csv" $2 1 > mlData/"mldrive${i}.txt.gran_1_results".txt
# done

# # For specified granularity
# if [ $3 -gt 1 ]
#    then
#       for i in 0 
#       do
#          python3 pred1.py \
#          mlData/"mldrive${i}_gran_${3}.csv" $2 $3 > mlData/"mldrive${i}.txt.gran_${3}_results".txt
#       done
# fi

cd mlData
mkdir -p drive0weights
cp mldrive0_gran_* drive0weights

cd ..
mkdir -p weights_header_1ssd

# For granularity == 1
python3 mlHeaderGen.py Trace sdb1 mlData/drive0weights weights_header_1ssd 1
if [ $3 -gt 1 ]
   then
      python3 mlHeaderGen.py Trace sdb1 mlData/drive0weights weights_header_1ssd $3
fi

cd ../kernel_hook/weights_header/mix
cp ../../../training/weights_header_1ssd/* .