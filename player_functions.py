import numpy as np
import math

total_matches = 10
cooperate = (3,0)
defect = (5,1)

class Bibble:
  def __init__(self, dna, fitness, cooperation, trustworthy):
    self.dna = dna
    self.fitness = fitness
    self.cooperation = cooperation
    self.trustworthy = trustworthy
    self.weight = self.precompute_weights()

  #Finds the weights of Bibble for respective clear score
  #ex) total_match = 5   => [f(0),f(1/5),f(2/5),f(3/5),f(4/5),f(5/5)]

  def precompute_weights(self):
    global total_matches
    c = self.dna[5]
    weights = []

    for i in range(total_matches+1):
      weights.append(impression_function(c, i/total_matches))
    return weights


def impression(bot,moves):
  global total_matches
  sample_size = len(moves)
  indices = np.linspace(0, total_matches, sample_size).astype(int)
  weights = [bot.weight[i] for i in indices]
  #Use the dot product to multiply the scores of each and add all up
  score = np.dot(weights,moves)
  impression_score = score / sum(weights)
  return impression_score


def geometric_mean(x):
    return (np.prod(x)+0.0001) ** (1 / len(x))

def impression_function(c,x):
  return 1-(math.tanh(math.pi*c*(1-x)))


def move_show(bot,my_moves,opp_moves):

  if my_moves == []:
    return bot.dna[0]

  agreeable = bot.dna[1]
  forgiving = bot.dna[2]
  apologetic = bot.dna[3]
  easy_going = bot.dna[4]
  immediate = bot.dna[5]

  impression_score = impression(bot,opp_moves)
  my_impression_score = impression(bot,my_moves)

  print("Impression Score ",impression_score)
  print("My Impression Score",my_impression_score)

  #Agreeable:
  agreeable_score = geometric_mean([impression_score,agreeable])
  print("Agreeable:",agreeable_score)

  #Forgiving
  forgiving_score = geometric_mean([impression_score,forgiving])
  print("Forgiving:",forgiving_score)


  #Apologetic
  apologetic_score = geometric_mean([1-my_impression_score,apologetic])
  print("Apologetic:",apologetic_score)

  #Easy Going
  #Introduce bias term to remove division by zero
  impression_diff = math.tanh((my_impression_score+0.1) * (impression_score+0.1) / ((my_impression_score+0.1)**2))
  easy_going_score = geometric_mean([impression_diff,easy_going])
  print("Easy-Going:",easy_going_score)

  # Give more weight to metrics depending on case
  if impression_score >= 0.5:
    final_score = (2*agreeable_score + 2*forgiving_score + apologetic_score + easy_going_score)/6
    print("Good Impression: ", final_score)

  else:
    final_score = (agreeable_score + forgiving_score + 2*apologetic_score + 2*easy_going_score)/6
    print("Bad Impression: ", final_score)

  return round(final_score)


def compete_show(bot1,bot2):
  global cooperate, defect
  bot1_moves = []
  bot2_moves = []


  for i in range(total_matches):
    move_1 = move_show(bot1, bot1_moves,bot2_moves)
    print()
    move_2 = move_show(bot2, bot2_moves,bot1_moves)
    if move_1 == 1 and move_2 == 1:
      bot1.fitness += cooperate[0]
      bot2.fitness += cooperate[0]
      bot1.cooperation += 1
      bot2.cooperation += 1
      bot1.trustworthy += 1
      bot2.trustworthy += 1

    elif move_1 == 1 and move_2 == 0:
      #Risk Factor for cooperating
      bot1.fitness += cooperate[1]
      bot2.fitness += defect[0]
      bot1.cooperation += 1

    elif move_1== 0 and move_2 == 1:
      bot1.fitness += defect[0]
      bot2.fitness += cooperate[1]
      bot2.cooperation += 1

    elif move_1 == 0 and move_2 == 0:
      bot1.fitness += defect[1]
      bot2.fitness += defect[1]

    bot1_moves.append(move_1)
    bot2_moves.append(move_2)
    print(bot1_moves)
    print(bot2_moves)
    print("\n\n")
  return bot1, bot2

def player_compete(bot1):
  global cooperate, defect, total_matches
  bot1_moves = []
  player_moves = []
  player_fitness = 0
  player_trustworthiness = 0
  player_cooperation = 0

  for i in range(total_matches):
    move_1 = move_show(bot1, bot1_moves,player_moves)
    print()
    move_2 = int(input("Enter 1 to cooperate, 0 to defect: "))
    if move_1 == 1 and move_2 == 1:
      bot1.fitness += cooperate[0]
      player_fitness += cooperate[0]
      bot1.cooperation += 1
      player_cooperation += 1
      bot1.trustworthy += 1
      player_trustworthiness += 1

    elif move_1 == 1 and move_2 == 0:
      #Risk Factor for cooperating
      bot1.fitness += cooperate[1]
      player_fitness += defect[0]
      bot1.cooperation += 1

    elif move_1== 0 and move_2 == 1:
      bot1.fitness += defect[0]
      player_fitness += cooperate[1]
      player_cooperation += 1

    elif move_1 == 0 and move_2 == 0:
      bot1.fitness += defect[1]
      player_fitness += defect[1]

    bot1_moves.append(move_1)
    player_moves.append(move_2)
    print("Bot:   ",bot1_moves)
    print("Player:",player_moves)
    print("Your Score:", player_fitness)
    print("Bot Score: ", bot1.fitness)
    print("Your Trustworthiness: ", player_trustworthiness)
    print("Bot Trustworthiness: ", bot1.trustworthy)
    print()
  return bot1





