# Test who generates the closer distribution in LCG and Matthews rotation

# 模拟掷骰子
def roll_dice():  
  roll = random.randint(1, 6)   # 按题目要求，使用randint在1~6范围取值
  return roll                   # 将结果返回
 
# 记录区
dice_tries1 = []
dice_tries2 = []
num_iterations = 100    # 次数
hits = 0   # 命中数
 
# 投掷次数
for i in range(num_iterations):
    dice_tries1.append(roll_dice())  # 投掷100次 存入数组
    dice_tries2.append(roll_dice())  # 投掷100次 存入数组
 
print("="*100)  
 
print(colored("* 红色表示两个骰子点数之和为8*", 'red'))   # 打印标题
 
change = 0
for j in range(num_iterations):
  if (change == 5):
      print()
      change = 0
  if (dice_tries1[j] + dice_tries2[j] == 8):
    print(colored("try %2s : %s %s" % (j, dice_tries1[j], dice_tries2[j]), "blue"), end = " ")
    change += 1
    hits += 1
  else:
    print("try %2s : %s %s" % (j, dice_tries1[j], dice_tries2[j]), end = " ")
    change += 1
 
 
print(colored("\n实际值 : 0.138889", "red"))
print(colored(f"计算值 : {round(hits / num_iterations,6)}", "red"))
print(colored(f"误差率 : {abs(hits / num_iterations - 5/36) / (5/36) * 100} %", "red"))
print("="*100)
