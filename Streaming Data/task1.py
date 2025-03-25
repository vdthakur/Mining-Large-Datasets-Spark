import binascii
import random
import math
from blackbox import BlackBox
import csv
import sys

# system arguments
input_f = sys.argv[1]
stream_s = int(sys.argv[2])
num_asks = int(sys.argv[3])
output_f = sys.argv[4]

# predefined bit array size of 69997, and number of hash functions set as 10
m = 69997
number_hash_functions = 10

# function to create a list of prime numbers
def create_primes(start_prime,stop_prime,count):
    # create an empty list to store prime numbers
    primes = []
    # for each number in the range of start and stop prime
    for possible_prime in range(start_prime,stop_prime):
        # set is_prime to True
        is_prime = True
        # for each number in the range of 2 to the square root of the possible prime plus 1
        for num in range(2, int(math.sqrt(possible_prime)) + 1):
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

# create a list of random values for a and b, set the range
a_hash = random.sample(range(m, 10 * m), number_hash_functions)
b_hash = random.sample(range(m, 10 * m), number_hash_functions)

# create a hash function using the formula (a * x + b) % p % m
def hash_function(a, b, p, m):
    return lambda x: ((a * x + b) % p) % m

# create a list of hash functions using the hash_function formula
hash_functions = [hash_function(a_hash[i], b_hash[i], prime_numbers[i], m) for i in range(number_hash_functions)]

# function to convert user to integer
def user_int(user):
    return int(binascii.hexlify(user.encode('utf8')), 16)

# function to create a list of hash values for a user
def myhashs(user_id):
    user_value = user_int(user_id)
    return [h(user_value) for h in hash_functions]

# initiate bit array with all values set to 0
bit_array = [0] * m
# initiate set to store previous users
previous_users_seen = set()

# create a black box object for the stream of data
bx = BlackBox()

# create a list to store the stream ask count and false positive rate
results = []
for ask in range(num_asks):
    # get the stream of data
    stream = bx.ask(input_f, stream_s)
    # reset the false positive and true negative counts to 0 at the start of each ask
    false_pos = 0
    true_negs = 0
    # for each user brought into the stream
    for user in stream:
        # create a list of hash values using previously defined hash functions
        hashes = myhashs(user)
        # use the hash values as an index and check if they are set to 1
        if all(bit_array[hash_value] == 1 for hash_value in hashes):
            # if all hash values are 1, check if user has been seen before
            if user not in previous_users_seen:
                # if user has not been seen before, increment the false positive count
                false_pos += 1
        else:
            # if not all hash values are 1, this is a true negative case
            true_negs += 1
        # use hash values as index and set values to 1 for the current user
        for hash_val in hashes:
            bit_array[hash_val] = 1
        # add user to the set of previous users
        previous_users_seen.add(user)
    
    # calculate the false positive rate
    total_negatives = false_pos + true_negs
    # edge case in the case FP + TN = 0 due to no more values in stream, set the false positive rate to 0
    fpr = false_pos / total_negatives if total_negatives > 0 else 0
    # append the ask and false positive rate to the results list
    results.append((ask,fpr))

with open(output_f, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["Time", "FPR"])
    for time, fpr in results:
        writer.writerow([time, fpr])