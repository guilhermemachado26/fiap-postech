import random
from game import simulate_game
from collections import deque
from shared_map import map

random.seed(42)

# Genetic algorithm definitions
num_individuals = 100
num_steps       = 100
num_generations = 1000
mutation_chance = 0.99
distance_weight = 0.99  # between 0 and 1 
directions = ['RIGHT', 'DOWN', 'LEFT', 'UP']
target = (8, 11)

# Mapping of directions to movements
movements = {
    'RIGHT': (0, 1),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'UP': (-1, 0)
}

class Player:
    def __init__(self, directions=None):
        if directions is None:
            directions = []
        self.directions = directions
        self.score = 0
        self.position = (1, 1)
        self.steps = 0

    def __repr__(self):
        return f"""
            Score: {self.score}           
            Positon: {self.position}      
            Steps: {self.steps}           
            Directions:
            {self.directions} 
            """

    def calculate_fitness(self):
        pos = (1, 1)  # Initial position
        self.score = 0
        x, y = (1, 1)
        for index, direction in enumerate(self.directions):
            mov = movements[direction]
            x, y = (x + mov[0], y + mov[1])
            self.steps = index +1

            # Hit the wall
            if map[x][y] == 1:
               # self.score -= 10
                break

            # Won the game
            if map[x][y] == 2:
                self.position = (x, y)
                self.score = 99999999999
                return
            
            # Did not hit the wall
            if map[x][y] == 0:
                self.position = (x, y)
                self.score += 1

        distance = calculate_distance(self.position, target)
        # Subtract distance from score to give direction
        self.score -= distance * distance_weight
        

def calculate_distance(start, goal):
    queue = deque([(start, 0)])
    visited = set()
    visited.add(start)

    while queue:
        (x, y), dist = queue.popleft()
        if (x, y) == goal:
            return dist
        
        for move in movements.values():
            new_x, new_y = x + move[0], y + move[1]
            if (0 <= new_x < len(map)) and (0 <= new_y < len(map[0])) and map[new_x][new_y] != 1 and (new_x, new_y) not in visited:
                queue.append(((new_x, new_y), dist + 1))
                visited.add((new_x, new_y))
    
    return float('-inf')  # Return -inf if no path

'''
 Create a player with 100 commands
'''
def create_individual():
    return Player([random.choice(directions) for _ in range(num_steps)])

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1.directions) - 1)
    child1 = Player(parent1.directions[:crossover_point] + parent2.directions[crossover_point:])
    child2 = Player(parent2.directions[:crossover_point] + parent1.directions[crossover_point:])
    return child1, child2

def mutate(individual):
    if random.random() < mutation_chance:
        mutation_point = random.randint(0, len(individual.directions) - 1)
        individual.directions[mutation_point] = random.choice(directions)
    return individual

# Genetic Algorithm
population = [create_individual() for _ in range(num_individuals)]

for generation in range(num_generations):
    for individual in population:
        individual.calculate_fitness()
    population = sorted(population, key=lambda x: x.score, reverse=True)

    if population[0].score >= 99999999999:  # Won
        break

    best_player = population[0]
    print("Generation ...:", generation)
    print("Best path found:\n", best_player)

    # cruzamento e mutacoes usando apenas os melhores individuos
    new_population = population[:num_individuals // 2]
    while len(new_population) < num_individuals:
        parent1, parent2 = random.sample(population[:num_individuals // 2], 2)
        child1, child2 = crossover(parent1, parent2)
        new_population.append(mutate(child1))
        new_population.append(mutate(child2))
    population = new_population
   
# Result
best_individual = population[0]
print("Best path found:", best_individual)
simulate_game(best_individual.directions)
