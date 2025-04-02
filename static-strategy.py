#Bibble Iteration #1: No attention to previous details

import random
import matplotlib.pyplot as plt

total_rounds = 20
total_generations = 25
population_size = 100

#Prisoner's Riddle for PHIL Youtube Video
# cooperate = (-1,-8)
# defect = (0,-3)

#Prisoner's Riddle --> Shifts back and forth
# cooperate = (3,-1)
# defect = (5,-3)

#Notice what happens when the risk is high for defecting
#This is why laws exist, if the reward is higher, but punishment will occur, agents choose cooperation
#Agents might defect occasionally in the hopes that the other might cooperate (e.g. Skipping the law)

cooperate = (3,-1)
defect = (6,-8)

#Nuclear War (Arms Race modeling)
# cooperate = (5,-20)
# defect = (3,-20)



class Bibble:
  def __init__(self, dna, fitness, cooperation):
    #DNA will be a binary array containing the moves that the algorithm plays as most optimal --> 20 matches
    self.dna = dna
    #Fitness: Total score that the bot accumulates, will later be used to determine fitness and reproduction algorithm
    self.fitness = fitness
    #Evaluates how many times the bot cooperates with other bots (High: Altruistic/ Low: Selfish)
    self.cooperation = cooperation

def compete(bot1,bot2):
  for i in range(len(bot1.dna)):

    cooperate_reward = []
    if bot1.dna[i] == 1 and bot2.dna[i] == 1:
      bot1.fitness += cooperate[0]
      bot2.fitness += cooperate[0]
      bot1.cooperation += 1
      bot2.cooperation += 1

    elif bot1.dna[i] == 1 and bot2.dna[i] == 0:
      #Risk Factor for cooperating
      bot1.fitness += cooperate[1]
      bot2.fitness += defect[0]
      bot1.cooperation += 1

    elif bot1.dna[i] == 0 and bot2.dna[i] == 1:
      bot1.fitness += defect[0]
      bot2.fitness += cooperate[1]
      bot2.cooperation += 1

    elif bot1.dna[i] == 0 and bot2.dna[i] == 0:
      bot1.fitness += defect[1]
      bot2.fitness += defect[1]

  return bot1, bot2


def random_dna():
  global total_rounds
  dna = []
  for i in range(total_rounds):
    dna.append(random.randint(0,1))
  return dna


def set_random_population(size):
  population = []
  for i in range(size):
    population.append(Bibble(random_dna(),0,0))
  return population

#Returns a random population of Bibbles
def population_war(population):
  for i in range(len(population)):
    for j in range(i+1,len(population)):
      compete(population[i],population[j])
  return population



def generation_results(population):
  global total_generations
  global original_fitness
  global original_cooperation

  rankings = dict()
  for i in range(len(population)):
    rankings[i] = (population[i].fitness, population[i].cooperation)

  rankings = dict(sorted(rankings.items(), key=lambda item: item[1], reverse=True))

  # for agent in rankings:
  #   print("Agent", agent, "| Fitness: ", rankings[agent][0], "| Cooperation: ",rankings[agent][1])

  fitness_val = []
  cooperation_val = []

  for i in rankings:
    fitness_val.append(rankings[i][0])
    cooperation_val.append(rankings[i][1])


  plt.title("Generation " + str(generation+1))
  plt.xlabel("Fitness")
  plt.ylabel("Cooperation")
  plt.scatter(original_fitness, original_cooperation, color='orange', label='Original')
  # Current population (blue)
  plt.scatter(fitness_val, cooperation_val, color='blue', label='New')
  plt.legend()
  plt.show()
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


def reproduce(bot1, bot2):
  global total_rounds
  child_dna = []
  for i in range(total_rounds):
    choice = random.randint(0,1)
    if choice == 0:
      child_dna.append(bot1.dna[i])
    else:
      child_dna.append(bot2.dna[i])

  child = Bibble(child_dna,0,0)
  return child

def mating_season(rankings,population):
  global population_size
  #Generate list of possibiities:
  new_gen = []
  new_gen += select_top_10(rankings, population)

  bibble_rankings= list(rankings.keys())

  reproduction_list = []
  for i in range(1,population_size + 1):
    reproduction_list += [i] * (100-i+1)
  random.shuffle(reproduction_list)

  for i in range(int(population_size*0.9)):
    bibble1 = bibble_rankings[random.randint(0,len(bibble_rankings)-1)]
    bibble2 = bibble_rankings[random.randint(0,len(bibble_rankings)-1)]
    new_gen.append(reproduce(population[bibble1],population[bibble2]))

  return new_gen


#Code for Generation Iterations
population = set_random_population(100)

for generation in range(total_generations):
  print("Generation", generation)
  print("_________________________")
  population = population_war(population)

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
  print("Total Cooperation Attempts", sum(population[i].dna), "| Fitness: ", population[i].fitness)
  print("Bot", i , "Algorithms: ",population[i].dna)
  print("___________________________-")






