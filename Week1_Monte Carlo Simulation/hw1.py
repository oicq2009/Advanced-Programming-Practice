// Plot log histogram (implemented using LCG)

import matplotlib.pyplot as plt
from tqdm import tqdm # optional
import random


# my LCG
def LCG(x, a, c, m):
  while True:
      x = (a * x + c) % m
      yield x


num_iterations = 100    # 次数
random_integers = []

def random_uniform_sample(n, space, seed=0):
  # Random seed
  a = 214013
  C = 2531011
  m = 2**32
  
  get = LCG(seed, a, C, m)
  lower, upper = space[0], space[1]

  # create the random value
  for i in range(n):
    value = (upper - lower) * (next(get) / (m - 1)) + lower
    random_integers.append(value)   

  return random_integers     # return the result


X = random_uniform_sample(num_iterations, [0, 99])
# print(X)

fig = plt.figure()
plt.hist(X)
plt.title(f"Check Uniform Distribution of {num_iterations} iterations")
plt.xlabel("Numbers")
plt.ylabel("N of each number")
plt.xlim([0, 99])
plt.show()
