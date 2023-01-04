'''
Prediction of victory in dice game
Players are playing dice, and the initial ownership cost is 1000 dollars.
Roll two dice. If the dice have the same number, you will get 4 dollars. If the dice have different numbers, you will lose 1 dollar
Use Monte Carlo algorithm to calculate the average rate of players according to the number of game executions, and confirm the average balance.
'''

import matplotlib.pyplot as plt
import random
 
def roll_dice():
  A = random.randint(1, 6)
  B = random.randint(1, 6)
  return A == B
 
# Inputs
num_simulations = 10000
max_num_rolls = 1000
bet = 1
 
win_probability = []
end_balance = []
 
 
plt.figure(figsize=(12,6))
 
for i in range(num_simulations):  # 投一万次骰子
  arr = [1000]
  rolls = [0]
  wins = 0
 
  while (rolls[-1] < max_num_rolls):
    if (roll_dice() == True):
      arr.append(arr[-1] + 4 * bet)
      wins += 1
    else:
      arr.append(arr[-1] - bet)
 
    rolls.append(rolls[-1] + 1)
 
  win_probability.append(wins / rolls[-1])
  end_balance.append(arr[-1])
  plt.plot(rolls, arr)
 
 
 
plt.title("Monte Carlo Dice Game [" + str(num_simulations) + " players simulations]")
plt.xlabel("Roll Number")
plt.ylabel("Balance [$]")
plt.xlim([0, max_num_rolls])
