
I. Monte Carlo simulation (Monte Carlo simulation)
0x00 Introduction: Concept description
💡 Introduction: In the stochastic model of a non-deterministic model, it is difficult to start with the analytical method, and then we can use "Monte Carlo simulation" (or Monte Carlo simulation) to solve the problem.

📚 Concept: "Monte Carlo simulation" is a computational algorithm that iteratively uses random sampling techniques to perform simulations and calculate the desired numerical results from the overall probability distribution.



 (Principality of Monaco - Monte Carlo)

Monte Carlo simulation can be useful in the fields of numerical integration, probability distribution calculation, probability-based optimization, etc. Example.

Simulation: To simulate a coin toss, take random values in the range.

Consider those less than 
Consider those greater than 
Monte Carlo method: Pouring a box of coins, counting the numbers and obtaining the probability.

Monte Carlo simulation: The probability is obtained by repeatedly extracting random values within a range. The Monte Carlo algorithm indicates that the more samples are taken, the more approximate the optimal solution.

🔍 Baidu encyclopedia: The encyclopedia gives an easy to understand example





0x01 Monte Carlo simulation routine model
🔺 It is divided into four steps as follows.

STEP1: Define the region (Range) to be sampled
STEP2: Random sampling of the defined area (Random sampling)
STEP3: Deterministic computation of the collected samples
STEP4: Statistical results and derivation of approximate values (Approximate)
💭 An example: two simple Monte Carlo simulation examples




STEP1: Generate two ranges of random numbers (roll the dice twice) and run the simulation for one random run
STEP2: If the sum of the two generated random numbers is , then , otherwise , count the number of and
STEP3: Calculate the ratio of the total number of executions to
STEP4: Compare with the mathematical probability

(taken from Wikipedia)

Calculate the circumference using a circle expressed as
The circle is contained in a square space of width 4 expressed as a circle.
The range is restricted to within
An ordered pair of random numbers is extracted from this space 
Among the extracted points, the number of points inside the circle is counted.
Use the ratio of the number of points inside the circle to the number of points in the whole to find 


II. Pseudo Random Number Generator (PRNG)
0x00 Introduction: Random number generation problem
In order to generate random numbers for Monte Carlo simulation, we need to place random number "seeds".

Here we choose to use a Pseudo Random Number Genarator (PRNG), for short .



In this chapter we will introduce the following classical pseudo random number generation algorithms: (we will focus on the first two algorithms)

Linear congruential generator (Linear congruential generator)
Mersenne Twister (Rotation algorithm)
John von Neumann mid-square method
multiplicative mid-product method
Constant multiplier method
Fibonacci Method
Lagged Fibonacci method
Shift Method
0x01 Linear congruence generator (LCG)
📚 Linear Congruential Generator (LCG), for short .

It generates segmented linear equations with a pseudo-random sequence of discontinuous computations, and the generator is defined by a cyclic relation as follows.



 is uniquely determined by the following parameters.

 , , , and   

📜 A sufficient condition necessary to maintain the maximum period of is as follows

 with mutual prime (coprime)
 is divisible by all prime factors of 
If is a multiple of then is also a multiple of
The parameters a, C, m are sensitive, they directly affect the quality of the pseudo-random number generation, so the values of the parameters are very important!


 It can generate random numbers with less memory, so it is suitable for embedded development environment.
It can be used without correlation, and is very simple to implement.

If the parameters and the last generated random number are known, all the factors generated afterwards can be predicted, so we cannot consider it a secure random number generator from a cryptographic point of view.
0x02 MersenneTwister rotation algorithm
"The MersenneTwister rotation algorithm is the best quality algorithm for generating random numbers today"

 The algorithm implements a good pseudo-random number, but the period is not long enough and it is easy for "bad people" to extrapolate random number seeds. Two scholars who had a good life - Matsumoto and Nishimura - worked on a new pseudo-random number algorithm in 1997.

The MersenneTwister rotation algorithm, or . It is based on linear recursion of matrices over finite binary fields. MersenneTwister generates high quality pseudo-random numbers quickly and corrects many defects of old random number generation algorithms. The algorithm uses Mersenne prime numbers, so it is also known as the Mersenne rotation algorithm. (However, most translations use "martenset" as the translation, so we will call it the martenset rotation algorithm here)



Even if you have never heard of this algorithm, you must have used the random module in Python: import random

import random
The random module in Python uses the Marteset rotation algorithm to implement random numbers, not only in Python, but also in PHP, Perl, and other popular programming languages, so the Marteset rotation algorithm is a very popular algorithm, and its status is obvious!

 Three stages of the algorithm.

Phase 1: Get the basic Matthiaset rotation chain
Phase 2: Rotation algorithm for the rotation chain
Phase 3: Processing of the results obtained from the rotation algorithm

The iterative period of the random number depends on the factor of the Materset.
Running on MT19937, the period can reach up to , with the property of 623 dimensional uniform distribution, which is really big! (Using 624 integers, 623*32=19936!) It has relatively small sequence correlations and can pass a variety of randomness tests, even Hdamard Test quantum calculations!
The algorithm, which can be implemented with only bitwise operations, is exceptionally fast.

The algorithm is a major drawback for capacity-constrained embedded environments because it must allocate space that can hold a number of digits.
From a cryptographic point of view, the Marteset rotation algorithm is still insecure, and when you know the properties of random numbers, it is possible to know the current state of the generator and predict the upcoming random numbers using only a limited number of random numbers (624).
0x03 Von Neumann Mid-square Method
The Mid-square Method is one of the first pseudo-random number generators, first proposed by von Neumann.

The meaning of the Mid-square Method is as its name implies.

Square: A 2s-bit decimal random number is squared to obtain a 4s-bit number (with the high bit complemented by 0 if it is less than 4s).
Squaring: it squares a 2s-bit random number and takes the middle 2s-bit number as a new random number.
This number is normalized (to a 2s-bit value less than 1) to the first random number, and so on, and the above process is repeated to obtain a series of pseudo-random numbers. The recursive formula is as follows.



Generating a pseudo-random number series.




Easy to implement, low memory consumption, simple computation.

It is difficult to specify what seed value can be taken to ensure a long enough period.
It is easy to have segment cycles with repeated elements, and it is easy to degenerate to a constant or even to zero, if an element degenerates to zero, then all subsequent elements will be zero.
📜 Variations and extensions of the squaring method.

Multiplicative mid-product method. 
To generate a sequence of pseudo-random numbers with decimal 2s bits, pick any two initial random numbers and recurse the formula.



To generate a pseudo-random sequence of numbers.



Constant multiplier method.
Recursive formula.



Generate a pseudo-random sequence of numbers.



0x04 Fibonacci Method
Unfortunately, this method is not a good generator of pseudo-random numbers. The method is based on the Fibonacci series, and its recursive formula is

     

This method has two initial seeds and has the great advantage that the computation cycle is fast and reaches full cycle. This generator has no multiplication operation and is generated very fast. The disadvantages are that the numbers in the sequence may recur, are less independent, and have intolerable uncentering.

📜 Deformations and extensions of the Fibonacci method (understanding).

Lagging Fibonacci Method.
Time lag Fibonacci generator, abbreviated as . The theory of time-lagged Fibonacci generators is quite complex, and the theory is not sufficient to guide the choice of the generator with . The initialization of the generator is also very sensitive, and its recursive formula is



where the new term is generated by the calculation of two old terms. Usually it is a power of 2 (), and often it is either . where the ★ operator represents the general binary operator, which can be addition, subtraction, multiplication or bitwise iso-or.

0x05 Shift Method
Computers are good at logical operations such as shifting, and the shift method uses this feature of computers to achieve this. It is an iterative process that combines shift operations with instruction summation operations. The method is as follows.

Take an initial value , so that it is shifted to the left and right, respectively, and then the instructions are added to obtain , and then the above process is repeated to produce a sequence of random numbers. The recursive formula is as follows (for 32-bit machines)



To generate a pseudo-random number sequence.



The shift method is fast, but it is very dependent on the initial value, which is generally not too small, and a poorly chosen initial value will make the pseudo-random number sequence short. At the same time, the independence crosses and the subsequent period of the random number sequence is related to the word length of the computer.

III. Python's random number generation function (Random module)
0x00 Introduction: Introducing the Random module
The Random module in Python is implemented using the Mathesit rotation algorithm, as we mentioned earlier.

It generates a bit-precision float with a period of .

Functions are provided to extract the distribution: uniform, normal, lognormal, negative exponential, gamma, beta 



 📚 The random module must be introduced before use.

import random
0x01 random - generates a random number between 0.0 and 1.0
random.random() # generate a random number among the real numbers between 0.1 and 1

📚 Function: Generate a random range of real numbers.

💬 Code demonstration: generates a random number between real numbers.

print(random.random())




0x02 uniform - generates a random number in the specified range
random.uniform(a, b) # Generate a random number between a and b

📚 Function: Generate a random real number in the specified range.

💬 Code demo: Generate a number between 100 and 300

print(random.uniform(10, 30))



0x03 randint - generates a random integer in the specified range
random.randint(a, b) # Generate a random integer between a and b

📚 Function: Generate a random integer in the specified range.

💬 Code demo: Generate an integer between 100 and 300

print(random.randint(100, 300))



0x04 choice - choose randomly in the sample set
random.choice(sample_set) # choose a random sample in the sample set sample_set

📚 Function: Select a random sample in the sample set

💬 Code demonstration: select a random sample from the sample set

L = [1, -3, 5, -2, 6, 8, -3, 0]  
print(random.choice(L))




0x05 sample - randomly select multiple times in the sample set
random.sample(sample_set, n) # randomly select n samples in the sample set sample_set

📚 Function: Execute "select a random sample in the sample set" n times.

💬 Code demonstration: randomly select multiple times from the sample set

L2 = [1, 'CSDN', 5, 'bilibili', 6, -3, 0]
print(random.sample(L2, 3)) # select 3 times from sample set L2
print(random.sample(L2, 6)) # Choose 6 times in sample set L2
print(random.sample(L2, 3)) # select 3 times in sample set L2



