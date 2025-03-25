from blackbox import BlackBox
import random
import csv
import sys

input_f = sys.argv[1]
stream_size = int(sys.argv[2])
num_of_asks = int(sys.argv[3])
output_f = sys.argv[4]

# define the reservoir size and create the blackbox object
reservoir_s = 100 
bx = BlackBox()

# initialize the reservoir, sequence number, and random seed
reservoir = [] 
sequence_number = 0 
random.seed(553)

# create a list of sequences
sequence_list = []
# for each ask
for ask in range(num_of_asks):
    # get the stream of data
    stream = bx.ask(input_f, stream_size)
    # for each user in the stream of data
    for user in stream:
        # increment sequence number by 1
        sequence_number += 1
        # if the number of sequences is less than or equal to the reservoir size
        if sequence_number <= reservoir_s:
            # append the user to the reservoir (first 100 will be appended because this statement will hold true for up to 100 sequences)
            reservoir.append(user)
        # else if the sequence number is greater than the reservoir size
        else:
            # for the nth user, decide whether to keep it or discard it using (reservoir size / number of sequences)
            probability = reservoir_s / sequence_number
            # if the random number is less than the probability, replace a random element in the reservoir
            if random.random() < probability:
                # replace a random element in the reservoir
                shift_index = random.randint(0,reservoir_s - 1)
                reservoir[shift_index] = user

    # append the sequence number, and the first, 20th, 40th, 60th, and 80th user to the sequence list
    sequence_list.append((sequence_number,reservoir[0],reservoir[20],reservoir[40],reservoir[60],reservoir[80]))


with open(output_f, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["seqnum","0_id","20_id","40_id","60_id","80_id"])
    writer.writerows(sequence_list)