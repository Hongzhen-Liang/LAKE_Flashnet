import csv
import sys
from enum import IntEnum

LEN_HIS_QUEUE = 4
IO_READ = '1'
IO_WRITE = '0'
LABEL_ISSUE = 'i'
LABEL_COMPLETION = 'c'


# this is how the replayer now outputs things
#        sprintf(buf, "%.3ld,%d,%d,%ld,%lu,%.3ld,%u,%lu", 
#   ts, latency, !op, 
#                   size, offset, submission, device, io_index);
#
class ReplayFields(IntEnum):
    TS = 0
    LATENCY = 1
    OP = 2
    SIZE = 3
    OFFSET = 4
    SUBMISSION = 5
    DEVICE = 6
    INDEX = 7

def generate_raw_vec(input_path, output_path, device_index):
    '''
        Traslate the `TrainTraceOutput_baseline.data` to `temp1`.
        @input_path: 
            Points to the file `TrainTraceOutput_baseline.data`.
            The features of a row is [TS, LATENCY, OP, SIZE, OFFSET, SUBMISSION, DEVICE]
        @output_path:
            Points to the file `temp1`.
            The features of a row are row[issue_i] = [hist_size_(compelete-4), hist_size_(compelete-3), hist_size_(compelete-2), hist_size_(compelete-1), IO_size(issue_i), hist_latency_(compelete-4), hist_latency_(compelete-3), hist_latency_(compelete-2), hist_latency_(compelete-1), hist_latency_(issue_i)]
            e.g. hist_size_(compelete-1) means the pending pages of the last complete request.
    '''
    with open(input_path, 'r') as input_file:
        input_csv = csv.reader(input_file)

        trace_list = []
        transaction_list = []
        index = 0
        for row in input_csv:
            if row[ReplayFields.DEVICE] != device_index:
                continue
            latency = int(row[ReplayFields.LATENCY])
            type_op = row[ReplayFields.OP]
            #size_ori = int(row[ReplayFields.SIZE])
            size = int((int(row[ReplayFields.SIZE])/512 + 7)/8)
            issue_ts = int(float(row[ReplayFields.SUBMISSION])) #this is in us now, so no *1000
            complete_ts = issue_ts+latency

            # trace_list.append([latency, type_op, size, issue_ts, complete_ts, 0])
            trace_list.append([size, type_op, latency, 0, index])  #history_queue.append([io[2], io[3]])
            transaction_list.append([index, issue_ts, LABEL_ISSUE])   # req issue
            transaction_list.append([index, complete_ts, LABEL_COMPLETION])  # req complete
            # index is used by trans to find corresponding trace_list
            index += 1

    #relying on stable sort, oof
    transaction_list = sorted(transaction_list, key=lambda x: x[1])   # sort by timestamp
    print('trace loading completed:', len(trace_list), 'samples')
    with open(output_path, 'w') as output_file:
        count = 0
        skip = 0
        pending_io = 0
        history_queue = [[0, 0]]*LEN_HIS_QUEUE  # 8 entries
        raw_vec = [0]*(LEN_HIS_QUEUE*2+1+1) # 10 entries
        # print(history_queue)

        # this is what this list looks like, but for some reason sorted by ts
        #  transaction_list.append([index, issue_ts, LABEL_ISSUE])
        #  transaction_list.append([index, complete_ts, LABEL_COMPLETION])
        # and this is trace_list
        # trace_list.append([size, type_op, latency, 0, index])
        # [10007, 1666302867994275, 'c']
        # [10008, 1666302867994319, 'i']
        # [10008, 1666302867994368, 'c']

        for trans in transaction_list:
            io = trace_list[trans[0]]
            #io is entry of trace_list, format [size in pages, type_op, latency, 0, index]
            if trans[2] == LABEL_ISSUE:
                #print("issue: ", trans)
                pending_io += io[0]  #add # pages to pending
                io[3] = pending_io  # save # of pending pages in  [3]
                # io becomes [size in pages, type_op, latency, pending_pages, index]

                # LEN_HIS_QUEUE = 4
                # raw_vec[issue_i] = [hist_size_(compelete-4), hist_size_(compelete-3), hist_size_(compelete-2), hist_size_(compelete-1), hist_size_(issue_i), hist_latency_(compelete-4), hist_latency_(compelete-3), hist_latency_(compelete-2), hist_latency_(compelete-1), hist_latency_(issue_i)]
                if io[1] == IO_READ and skip >= LEN_HIS_QUEUE:  #start doing this after 4th
                    #print("actually apending now")
                    count += 1
                    raw_vec[LEN_HIS_QUEUE] = io[3]  #pending pages
                    raw_vec[-1] = io[2]  #latency
                    for i in range(LEN_HIS_QUEUE):
                        raw_vec[i] = history_queue[i][1]     # pending_bytes
                        raw_vec[i+LEN_HIS_QUEUE+1] = history_queue[i][0]    # latency
                    output_file.write(','.join(str(x) for x in raw_vec)+'\n')

            elif trans[2] == LABEL_COMPLETION:
                #print("complete: ", trans)
                #decrement pending bytes since one complete
                pending_io -= io[0]

                if io[1] == IO_READ:
                    history_queue.append([io[2], io[3]]) #append latency,pending_bytes
                    del history_queue[0]
                    skip += 1

        # print(history_queue)
        print(pending_io)
        print('Done:', count, 'vectors')
        print('wrote to ', output_path)


def rearrange_feats(granularity, input_path, output_path):
    '''
        Called after generated `temp1` by generate_raw_vec().
        @granularity:
            The number of IOs to be grouped together.
        @input_path:
            `temp1` which has the rows structure:
                [hist_size_(compelete-4), hist_size_(compelete-3), hist_size_(compelete-2), hist_size_(compelete-1), IO_size(1), hist_latency_(compelete-4), hist_latency_(compelete-3), hist_latency_(compelete-2), hist_latency_(compelete-1), hist_latency_(1)]
                [hist_size_(compelete'-4), hist_size_(compelete'-3), hist_size_(compelete'-2), hist_size_(compelete'-1), IO_size(2), hist_latency_(compelete'-4), hist_latency_(compelete'-3), hist_latency_(compelete'-2), hist_latency_(compelete'-1), hist_latency_(2)]
                [hist_size_(compelete''-4), hist_size_(compelete''-3), hist_size_(compelete''-2), hist_size_(compelete''-1), IO_size(3), hist_latency_(compelete''-4), hist_latency_(compelete''-3), hist_latency_(compelete''-2), hist_latency_(compelete''-1), hist_latency_(3)]
                [hist_size_(compelete```-4), hist_size_(compelete```-3), hist_size_(compelete```-2), hist_size_(compelete```-1), IO_size(4), hist_latency_(compelete```-4), hist_latency_(compelete```-3), hist_latency_(compelete```-2), hist_latency_(compelete```-1), hist_latency_(4)]
                [hist_size_(compelete''''-4), hist_size_(compelete''''-3), hist_size_(compelete''''-2), hist_size_(compelete''''-1), IO_size(5), hist_latency_(compelete''''-4), hist_latency_(compelete''''-3), hist_latency_(compelete''''-2), hist_latency_(compelete''''-1), hist_latency_(5)]
                ...
        @output_path:
            group rows of `temp1` together according to @granularity. If @granularity = 4, the output rows structure will be:
                [hist_size_(compelete```-4), hist_size_(compelete```-3), hist_size_(compelete```-2), hist_size_(compelete```-1), IO_size(1), IO_size(2), IO_size(3), IO_size(4), hist_latency_(compelete```-4), hist_latency_(compelete```-3), hist_latency_(compelete```-2), hist_latency_(compelete```-1), hist_latency_(1), hist_latency_(2), hist_latency_(3), hist_latency_(4)]
                [hist_size_(compelete''''-4), hist_size_(compelete''''-3), hist_size_(compelete''''-2), hist_size_(compelete''''-1), IO_size(2), IO_size(3), IO_size(4), IO_size(5), hist_latency_(compelete''''-4), hist_latency_(compelete''''-3), hist_latency_(compelete''''-2), hist_latency_(compelete''''-1), hist_latency_(2), hist_latency_(3), hist_latency_(4), hist_latency_(5)]
                ...
    '''

    input_rows = []
    count = 0
    with open(input_path, 'r') as input_f:
        for line in input_f:
            input_rows.append(line.strip("\n").split(","))

    with open(output_path, 'w') as output_f:
        for i in range(granularity - 1, len(input_rows)):
            hist_size_list = [0] * LEN_HIS_QUEUE
            hist_latency_list = [0] * LEN_HIS_QUEUE
            IO_size_list = [0] * granularity
            latency_list = [0] * granularity

            current_io = input_rows[i]
            for j in range(LEN_HIS_QUEUE):
                hist_size_list[j] = current_io[j]
                hist_latency_list[j] = current_io[j + LEN_HIS_QUEUE + 1]

            for hist_index in range(granularity):
                IO_size_list[hist_index] = input_rows[i - granularity + 1 + hist_index][LEN_HIS_QUEUE]
                latency_list[hist_index] = input_rows[i - granularity + 1 + hist_index][-1]

            new_feat_row = hist_size_list + IO_size_list + hist_latency_list + latency_list
            output_f.write(','.join(str(x) for x in new_feat_row)+'\n')
            count += 1

        print('Done:', count, 'vectors')
        print('wrote to ', output_path)


def generate_ml_vec(granularity, len_pending, len_latency, input_path, output_path):
    '''
        Translate the `temp1` to `mldrive0.csv`
        @granularity:
            The number of IOs to be grouped together.
        @input_path:
            `temp1` file generated by `generate_raw_vec()`.
            The features are: row[issue_i] = [hist_size_(compelete-4), hist_size_(compelete-3), hist_size_(compelete-2), hist_size_(compelete-1), hist_size_(issue_i), hist_latency_(compelete-4), hist_latency_(compelete-3), hist_latency_(compelete-2), hist_latency_(compelete-1), hist_latency_(issue_i)]
        @output_path:
            'mlData/mldrive0.csv'
            The features are the same as `temp1`, but digits of each feature are seperated.
            e.g. temp1 has one row: ['11', '7', '8', '8', '8', '3', '4', '3', '1', '0']
                the input row to mldrive0.csv will be 0,1,1, 0,0,7, 0,0,8, 0,0,8, 0,0,8, 0,0,0,3, 0,0,0,4, 0,0,0,3, 0,0,0,1, 0
    '''
    count = 0
    max_pending = (10**len_pending)-1  # = 999
    max_latency = (10**len_latency)-1  # = 9999
    # print(max_pending, max_latency)  
    with open(input_path, 'r') as input_file, open(output_path, 'w') as output_file:
        input_csv = csv.reader(input_file)
        #5,4,4,9,15,186,132,143,51,54
        #9,5,10,15,4,51,189,75,54,157
        for rvec in input_csv:
            tmp_vec = []
            for i in range(LEN_HIS_QUEUE+granularity):  # LEN_HIS_QUEUE = 4
                pending_io = int(rvec[i])    # [hist_size_(compelete-4), hist_size_(compelete-3), hist_size_(compelete-2), hist_size_(compelete-1), hist_size_(issue_i)]
                if pending_io > max_pending:
                    pending_io = max_pending
                tmp_vec.append(','.join(x for x in str(pending_io).rjust(len_pending, '0')))
            for i in range(LEN_HIS_QUEUE):
                latency = int(rvec[i+LEN_HIS_QUEUE+granularity])  # [hist_latency_(compelete-4), hist_latency_(compelete-3), hist_latency_(compelete-2), hist_latency_(compelete-1)]
                if latency > max_latency:
                    latency = max_latency
                tmp_vec.append(','.join(x for x in str(latency).rjust(len_latency, '0')))
            for i in range(granularity):
                tmp_vec.append(rvec[-granularity + i]) # [hist_latency_(issue_i)]
            output_file.write(','.join(x for x in tmp_vec)+'\n')
            count += 1
            # print("writing ", ','.join(x for x in tmp_vec)+'\n')

    print(f"wrote {count} to ", output_path)

if len(sys.argv) < 2:
    print('illegal cmd format')
    exit(1)

mode = sys.argv[1]
if mode == 'raw':
    if len(sys.argv) != 4:
        print('illegal cmd format')
        exit(1)
    trace_path = sys.argv[2]
    raw_path = sys.argv[3]
    generate_raw_vec(trace_path, raw_path)
elif mode == 'ml':
    if len(sys.argv) != 6:
        print('illegal cmd format')
        exit(1)
    len_pending = int(sys.argv[2])
    len_latency = int(sys.argv[3])
    raw_path = sys.argv[4]
    ml_path = sys.argv[5]
    generate_ml_vec(len_pending, len_latency, raw_path, ml_path)
elif mode == 'direct':
    #only modified
    if len(sys.argv) != 9:
        print('illegal cmd format')
        exit(1)
    len_pending = int(sys.argv[2])   # = 3
    len_latency = int(sys.argv[3])   # = 4
    trace_path = sys.argv[4]  #mlData/TrainTraceOutput
    temp_file_path = sys.argv[5]  #mlData/temp1
    temp_file_rearrange_path = temp_file_path + '_rearrange'
    output_path = sys.argv[6] #mlData/"mldrive${i}.csv"
    device_index = sys.argv[7] #"$i"
    granularity = int(sys.argv[8])   # = 4

    generate_raw_vec(trace_path, temp_file_path, device_index)
    rearrange_feats(granularity, temp_file_path, temp_file_rearrange_path)
    generate_ml_vec(granularity, len_pending, len_latency, temp_file_rearrange_path, output_path)
else:
    print('illegal mode code')
    exit(1)

# trace = 'WD_NVMe_1_6T.bingselection.drive0.rr1.exp_0'
# trace_path = '/Users/linanqinqin/Documents/DESS/Macrobench/replayML/'+trace+'.csv'
# raw_path = '/Users/linanqinqin/Documents/DESS/Macrobench/replayML/'+trace+'.raw_vec.csv'
# ml_path = '/Users/linanqinqin/Documents/DESS/Macrobench/replayML/'+trace+'.ml_vec.31.csv'
# generate_raw_vec(trace_path, raw_path)
# generate_ml_vec(3, 4, raw_path, ml_path)
