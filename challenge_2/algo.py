"""
Genetic Algorithm to solve the maze game.
Each player has a list of commands to follow.

The fitness score is calculated as the Manhattan distance to the goal.
If the player hits a wall, the player is penalized with a negative score.
The maze exit is represented by the number 2 in the maze map.
"""

import random
from game import simulate_game

class Player:
    def __init__(self, directions=[]):
        self.directions = directions
        self.score = 0

    def __repr__(self):
        return f"Directions: {self.directions} - Score: {self.score}"
    

class Maze:
    def __init__(self):
        self.start = (1, 1)
        self.exit = (8, 11)
        self.maze_map = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    
    def get_final_position(self, commands):
        x, y = self.start
        for command in commands:
            if command == "UP":
                x -= 1
            elif command == "DOWN":
                x += 1
            elif command == "LEFT":
                y -= 1
            elif command == "RIGHT":
                y += 1

            if self.maze_map[x][y] == 1:
                return None
        return x, y


class GenMaze:
    def __init__(self, maze):
        self.maze = maze
        self.players = []

    def initialize_population(self, size=100, length=2):
        """
        Initialize the population with random players with random commands of length 2.
        """
        self.players = [
            Player(random.sample(["UP", "DOWN", "LEFT", "RIGHT"], k=length))
            for _ in range(size)
        ]

    def calculate_score(self, player):
        """
        Calculate the fitness score of a player.
        More close to the goal, the better.
        If the player hits a wall, the player is penalized with a negative score.
        """
        final_position = self.maze.get_final_position(player.directions)
        
        if final_position is None:
            return -1000

        # Calculate Manhattan distance to the goal
        distance_to_goal = abs(final_position[0] - self.maze.exit[0]) + abs(final_position[1] - self.maze.exit[1])

        # Fitness score: the closer to the goal, the better
        return max(0, 1000 - distance_to_goal * distance_to_goal)

    def selection(self):
        """
        Use tournament selection to select the best player.
        """
        tournament = random.sample(self.players, k=10)
        return max(tournament, key=lambda x: x.score)
    
    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1.directions) - 1)

        child_directions = parent1.directions[:crossover_point] + parent2.directions[crossover_point:]
        child = Player(directions=child_directions)
        child.score = self.calculate_score(child)

        return child
    
    def mutate(self, player, mutation_rate=0.8):
        """
        Randomly mutate the player's commands.
        It also consider adding a new command to extend the player's path.
        """
        for i in range(len(player.directions)):
            if random.random() < mutation_rate:
                player.directions[i] = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])

        if random.random() < mutation_rate:
            player.directions.append(random.choice(["UP", "DOWN", "LEFT", "RIGHT"]))

    def replace_population(self, old_population, new_population):
        """
        Union of the old population and the new population.
        Returns the new population with the best players.
        """
        return sorted(old_population + new_population, key=lambda x: x.score, reverse=True)[:len(old_population)]
    
    def run(self, generations, population_size, path_length):
        self.initialize_population(population_size, path_length)
        print(f"Polulation initialized with {len(self.players)} players.")
    
        for generation in range(generations):
            print(f"Running generation {generation}...")

            for player in self.players:
                player.score = self.calculate_score(player)

            new_players = []
            
            for _ in range(population_size):
                parent1 = self.selection()
                parent2 = self.selection()
                child = self.crossover(parent1, parent2)
                
                self.mutate(child)
                child.score = self.calculate_score(child)

                new_players.append(child)

            self.players = self.replace_population(self.players, new_players)
            best_player = max(self.players, key=lambda x: x.score)
            print(f"Best player: {best_player}")

        best_player = max(self.players, key=lambda x: x.score)

        print(best_player)
        simulate_game(best_player.directions)
        return best_player


if __name__ == "__main__":
    gen = GenMaze(maze=Maze())
    gen.run(generations=10000, population_size=100, path_length=2)
