from blackbox import BlackBox
import binascii
import random
import math
import csv
import sys

input_f = sys.argv[1]
stream_size = int(sys.argv[2])
num_of_asks = int(sys.argv[3])
output_f = sys.argv[4]

# define the number of hash functions, group size, number of groups
number_hash_functions = 100
group_size = 10
number_groups = number_hash_functions // group_size

# initialize a value of m to define hash range and create the blackbox object
# this value was experimented with and set to 1000 to achieve a better score (estimates/groundtruth)
m = 1000
bx = BlackBox()

# function to create a list of prime numbers
def create_primes(start_prime,stop_prime,count):
    # create an empty list to store prime numbers
    primes = []
    # for each number in the range of start and stop prime
    for possible_prime in range(start_prime, stop_prime):
        # set is_prime to True
        is_prime = True
        # for each number in the range of 2 to the square root of the possible prime plus 1
        for num in range(2,int(math.sqrt(possible_prime)) + 1):
            # if the possible prime is divisible by i, set is_prime to False and break the loop
            if possible_prime % num == 0:
                is_prime = False
                break
        # if is_prime is still True, append the possible prime to the list of primes
        if is_prime:
            primes.append(possible_prime)
            if len(primes) == count:
                break
    # return the list of primes
    return primes

# create a list of prime numbers between large prime numbers (10m and 20m)
prime_range_start = m * 10
prime_range_end = m * 20
# store the list of prime numbers in prime_numbers
prime_numbers = create_primes(prime_range_start, prime_range_end, number_hash_functions)

# create a list of random values for a and b with large range
a_hash = random.sample(range(m, 10 * m), number_hash_functions)
b_hash = random.sample(range(m, 10 * m), number_hash_functions)

# create a hash function using the formula (a * x + b) % p % m
def hash_function(a, b, p, m):
    return lambda x: ((a * x + b) % p) % m

# create a list of hash functions using the hash_function formula
hash_functions = [hash_function(a_hash[i], b_hash[i], prime_numbers[i], m) for i in range(number_hash_functions)]

def user_integer(user):
    return int(binascii.hexlify(user.encode('utf8')), 16)

def myhashs(user_id):
    user_value = user_integer(user_id)
    return [h(user_value) for h in hash_functions]

# Count trailing zeros in the binary representation
def count_zeros(value):
    binary_representation = bin(value)[2:]  # Get binary string
    return len(binary_representation) - len(binary_representation.rstrip('0'))

# process the streams
# initate a list to store the estimates
estimates = []
# for each ask
for i in range(num_of_asks):
    # get a stream of data
    stream = bx.ask(input_f, stream_size)
    # initialize max trailing zeros for each hash function
    maximum_trailing_zeros = [0] * number_hash_functions
    
    # Update max trailing zeros for each user in the stream
    for user in stream:
        # compute hash values for the user
        hash_values = myhashs(user)
        # for each hash value, update the max trailing zeros (all are set to 0 initially)
        for i, hash_value in enumerate(hash_values):
            maximum_trailing_zeros[i] = max(maximum_trailing_zeros[i], count_zeros(hash_value))
    
    # compute the estimates in a group wise manner
    grp_estimates = []
    # for each group in the number of groups
    for group in range(number_groups):
        # get the portion of `max_trailing_zeros` corresponding to the current group by slicing to select `group_size` elements starting from `group * group_size` to the next `group_size` elements
        group_hashed = maximum_trailing_zeros[group * group_size:(group + 1) * group_size]
        # compute the group estimate as 2 raised to the power of the average of the group hashes
        grp_est = 2 ** (sum(group_hashed) / group_size)
        # append the group estimate to the list of group estimates
        grp_estimates.append(grp_est)  
    
    # find the median of the group estimates as the final estimate
    final_estimate = sorted(grp_estimates)[len(grp_estimates) // 2]
    # retrieve the ground truth
    ground_truth = len(stream)
    # append the ground truth and final estimate to the list of estimates
    estimates.append((ground_truth,int(final_estimate)))

# code to determine whether task passes evaluation criteria
# truth_sum = 0
# estimate_sum = 0
# for i, estimate in estimates:
#     truth_sum += i
#     estimate_sum += estimate
# result = estimate_sum / truth_sum
# print(f"Evaluation = {result}")

# write results to the output file
with open(output_f, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Time','Ground Truth','Estimate'])
    for index, (ground_truth, estimate) in enumerate(estimates):
        writer.writerow([index, ground_truth, estimate])