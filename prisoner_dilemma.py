from typing_extensions import final
#New Attempt

import random
import matplotlib.pyplot as plt
import math
import random
import numpy as np
from multiprocessing import Pool

#Settings
# total_matches = 20
# total_generations = 50
# population_size = 100

# cooperate = (3,0)
# defect = (5,1)


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

def random_dna():
  global total_rounds
  dna = []
  #Initial
  dna.append(random.randint(0,1))
  #Agreeable
  dna.append(random.uniform(0, 1))
  #Forgiveness
  dna.append(random.uniform(0, 1))
  #Apologetic
  dna.append(random.uniform(0, 1))
  #Easy Going
  dna.append(random.uniform(0, 1))
  #Clear/Memory
  dna.append(random.uniform(0, 1))

  return dna



def move(bot,my_moves,opp_moves):

  if my_moves == [] or opp_moves == 0:
    return bot.dna[0]

  agreeable = bot.dna[1]
  forgiving = bot.dna[2]
  apologetic = bot.dna[3]
  easy_going = bot.dna[4]
  immediate = bot.dna[5]

  impression_score = impression(bot,opp_moves)
  my_impression_score = impression(bot,my_moves)

  #Agreeable:
  agreeable_score = geometric_mean([impression_score,agreeable])

  #Forgiving
  forgiving_score = geometric_mean([impression_score,forgiving])

  #Apologetic
  apologetic_score = geometric_mean([1-my_impression_score,apologetic])

  #Easy Going
  #Introduce bias term to remove division by zero
  impression_diff = math.tanh((my_impression_score+0.1) * (impression_score+0.1) / ((my_impression_score+0.1)**2))
  impression_diff = math.tanh((my_impression_score + 0.1) * (impression_score + 0.1) / ((my_impression_score + 0.1) ** 2))

  easy_going_score = geometric_mean([impression_diff,easy_going])

  # Give more weight to metrics depending on case
  if impression_score >= 0.5:
    final_score = (2*agreeable_score + 2*forgiving_score + apologetic_score + easy_going_score)/6

  else:
    final_score = (agreeable_score + forgiving_score + 2*apologetic_score + 2*easy_going_score)/6

  return round(final_score)


def compete(bot1,bot2):
  global cooperate, defect, total_matches
  bot1_moves = []
  bot2_moves = []


  for i in range(total_matches):
    move_1 = move(bot1, bot1_moves,bot2_moves)
    move_2 = move(bot2, bot2_moves,bot1_moves)
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

  return bot1, bot2


def set_random_population(size):
  population = []
  for i in range(size):
    population.append(Bibble(random_dna(),0,0,0))
  return population



def compete_wrapper(args):
    """Wrapper to unpack arguments for `compete`."""
    bot1, bot2 = args
    return compete(bot1, bot2)


def population_war(population):
  for i in range(len(population)):
    for j in range(i+1,len(population)):
      compete(population[i],population[j])
  return population


def generation_results(population,generation):
  global total_generations
  global original_fitness
  global original_cooperation

  rankings = dict()
  for i in range(len(population)):
    rankings[i] = (population[i].fitness, population[i].cooperation)

  rankings = dict(sorted(rankings.items(), key=lambda item: item[1], reverse=True))

  for agent in rankings:
    print("Agent", agent, "| Fitness: ", rankings[agent][0], "| Cooperation: ",rankings[agent][1])

  fitness_val = []
  cooperation_val = []

  for i in rankings:
    fitness_val.append(rankings[i][0])
    cooperation_val.append(rankings[i][1])


  plt.title("Generation " + str(generation))
  plt.xlabel("Fitness")
  plt.ylabel("Cooperation")
  # plt.scatter(original_fitness,original_cooperation)
  plt.scatter(fitness_val,cooperation_val)
  plt.show()
  return rankings


  # for agent in rankings:
  #   print("Agent", agent, "| Fitness: ", rankings[agent][0], "| Cooperation: ",rankings[agent][1])



def generation_results(population):
  global total_generations
  global original_fitness
  global original_cooperation

  rankings = dict()
  for i in range(len(population)):
    rankings[i] = (population[i].fitness, population[i].cooperation)

  rankings = dict(sorted(rankings.items(), key=lambda item: item[1][0], reverse=True))


  # for agent in rankings:
  #   print("Agent", agent, "| Fitness: ", rankings[agent][0], "| Cooperation: ",rankings[agent][1])

  fitness_val = []
  cooperation_val = []

  for i in rankings:
    fitness_val.append(rankings[i][0])
    cooperation_val.append(rankings[i][1])


  plt.title("Generation " + str(generation))
  plt.xlabel("Fitness")
  plt.ylabel("Cooperation")
  plt.scatter(original_fitness, original_cooperation, color='orange', label='Original')
  # Current population (blue)
  plt.scatter(fitness_val, cooperation_val, color='blue', label='New')
  plt.legend()
  plt.show()

  bibble_rankings= list(rankings.keys())
  return rankings


def select_top_10(rankings, population):
  global population_size
  top_10 = []
  #top 10 percent are left
  for i in range(int(population_size*0.1)):
    bibble = population[list(rankings.keys())[i]]
    bibble.fitness = 0
    bibble.cooperation = 0
    top_10.append(bibble)
  return top_10


# def reproduce(bot1, bot2):
#   global total_rounds
#   child_dna = []
#   if bot1.dna[0] == bot2.dna[0]:
#     child_dna.append(bot1.dna[0])
#   else:
#     child_dna.append(random.randint(0,1))

#   for i in range(1,len(bot1.dna)):
#     child_dna.append((bot1.dna[i] + bot2.dna[i])/2)

#   child = Bibble(child_dna,0,0,0)
#   return child


def reproduce(bot1, bot2, mutation_rate=0.3, mutation_strength=0.15):
    """
    Reproduces a child Bibble with a mix of bot1 and bot2 DNA,
    and applies random mutations while ensuring DNA values remain within [0, 1].

    Args:
    - bot1 (Bibble): First parent.
    - bot2 (Bibble): Second parent.
    - mutation_rate (float): Probability of mutation for each gene.
    - mutation_strength (float): Maximum deviation for mutation.

    Returns:
    - Bibble: Child Bibble with mutated DNA.
    """
    child_dna = []

    # Combine DNA from both parents for the first gene (binary decision)
    if bot1.dna[0] == bot2.dna[0]:
        child_dna.append(bot1.dna[0])
    else:
        child_dna.append(random.randint(0, 1))

    # For other genes, mix parent DNA and apply mutation
    for i in range(1, len(bot1.dna)):
        # Average the DNA from both parents
        gene = (bot1.dna[i] + bot2.dna[i]) / 2

        # Apply mutation with a probability of `mutation_rate`
        if random.random() < mutation_rate:
            mutation = random.uniform(-mutation_strength, mutation_strength)
            gene += mutation

        # Ensure the gene value stays within [0, 1]
        gene = max(0, min(1, gene))
        child_dna.append(gene)

    # Create and return the child Bibble
    child = Bibble(child_dna, 0, 0, 0)
    return child


# def freaky_time(rankings,population):
#   global population_size
#   #Generate list of possibiities:
#   new_gen = []
#   new_gen += select_top_10(rankings, population)
#   attraction_weights = [len(rankings) - rank for rank in range(len(rankings))]
#   print(attraction_weights)
#   bibble_rankings= list(rankings.keys())

#   for i in range(int(population_size*0.9)):
#       # Select parents with weights
#       bibble1 = population[random.choices(bibble_rankings, weights=attraction_weights)[-1]]
#       bibble2 = population[random.choices(bibble_rankings, weights=attraction_weights)[-1]]
#       print(bibble1.fitness,bibble2.fitness)
#       # Reproduce and add to the new generation
#       child = reproduce(bibble1, bibble2)
#       new_gen.append(child)

#   return new_gen

def mating_season(rankings,population):
  global population_size
  #Generate list of possibiities:
  new_gen = []
  new_gen += select_top_10(rankings, population)

  bibble_rankings= list(rankings.keys())

  reproduction_list = []
  for i in range(population_size):
    reproduction_list += [bibble_rankings[i]] * (population_size-i+1)

  random.shuffle(reproduction_list)

  for i in range(int(population_size*0.9)):
    bibble1 = bibble_rankings[random.randint(0,len(bibble_rankings)-1)]
    bibble2 = bibble_rankings[random.randint(0,len(bibble_rankings)-1)]
    new_gen.append(reproduce(population[bibble1],population[bibble2]))

  return new_gen


total_matches = 20
total_generations = 20
population_size = 100


cooperate = (3,0)
defect = (5,1)


population = set_random_population(population_size)

for generation in range(total_generations):
  print("Generation", generation)
  print("_________________________")

  population = population_war(population)
  print(len(population))
  if generation == 0:
    original_fitness = [agent.fitness for agent in population]
    original_cooperation = [agent.cooperation for agent in population]

  rankings = generation_results(population)
  new_gen = mating_season(rankings,population)
  population = new_gen

print("\n\n\n\n")

print("Final Generation")
print("_________________________")
population = population_war(population)
rankings = generation_results(population)

for i in rankings:
  print("Total Cooperation Attempts", sum(population[i].dna), "| Fitness: ", population[i].fitness, "| Trustworthiness: ", population[i].trustworthy)
  print("Bot", i , "Algorithms: ",population[i].dna)
  print("___________________________-")

print("Population Size : ", len(population))