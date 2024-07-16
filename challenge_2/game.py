import pygame
from shared_map import map

def simulate_game(commands=None):
    pygame.init()
    screen = pygame.display.set_mode((700, 700))
    pygame.display.set_caption("Game Simulation")

    speed = 50
    x, y = 50, 50

    # Load the images with transparency
    person_image = pygame.image.load('./person.png').convert_alpha()
    person_image = pygame.transform.scale(person_image, (50, 50))  # Resize if necessary
    exit_image = pygame.image.load('./door.png').convert_alpha()
    exit_image = pygame.transform.scale(exit_image, (50, 50))  # Resize if necessary

    def game_map():
        goal_surface = pygame.Surface((80, 80), pygame.SRCALPHA)
        pygame.draw.rect(goal_surface, (255, 255, 0), (0, 0, 50, 50))
    
        wall_surface = pygame.Surface((100, 100), pygame.SRCALPHA)
        pygame.draw.rect(wall_surface, (0, 0, 0), (0, 0, 50, 50)) 

        path_surface = pygame.Surface((80, 80), pygame.SRCALPHA)
        pygame.draw.rect(path_surface, (255, 255, 255), (0, 0, 50, 50)) 

        mapping = {
            0: path_surface,
            1: wall_surface,
            2: exit_image,  # Use the exit_image for the goal
        }

        for y, row in enumerate(map):
            for x, cell in enumerate(row):
                image = mapping[cell]
                screen.blit(image, [x*50, y*50])

        pygame.display.update() 

    def player(x, y):
        screen.blit(person_image, (x, y))


 # Desenhe o mapa e o jogador antes do loop principal
    screen.fill((255, 255, 255))
    game_map()
    player(x, y)
    pygame.display.update()
    
    loop = True
    clock = pygame.time.Clock()

    command_index = 0
    while loop:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                loop = False
            elif event.type == pygame.KEYDOWN:
                if commands is None:  # Only handle key events if no commands are provided
                    if event.key == pygame.K_LEFT:
                        command = 'LEFT'
                    elif event.key == pygame.K_RIGHT:
                        command = 'RIGHT'
                    elif event.key == pygame.K_UP:
                        command = 'UP'
                    elif event.key == pygame.K_DOWN:
                        command = 'DOWN'
                    else:
                        continue

                    pos = x, y
                    if command == 'LEFT':
                        x -= speed
                    elif command == 'RIGHT':
                        x += speed
                    elif command == 'UP':
                        y -= speed
                    elif command == 'DOWN':
                        y += speed

                    row = y // 50
                    column = x // 50
                    if map[row][column] == 1:
                        x, y = pos
                        print('Hit the wall')
                    if map[row][column] == 2:
                        print('You Win')
                        loop = False

                    screen.fill((255, 255, 255))
                    game_map()
                    player(x, y)
                    pygame.display.update()
                    clock.tick(10)
        
        if commands and command_index < len(commands):
            command = commands[command_index]
            command_index += 1

            pos = x, y
            if command == 'LEFT':
                x -= speed
            elif command == 'RIGHT':
                x += speed
            elif command == 'UP':
                y -= speed
            elif command == 'DOWN':
                y += speed

            row = y // 50
            column = x // 50
            if map[row][column] == 1:
                x, y = pos
                print('Hit the wall')
            if map[row][column] == 2:
                print('You Win')
                loop = False

            screen.fill((255, 255, 255))
            game_map()
            player(x, y)
            pygame.display.update()
            clock.tick(10)

    pygame.quit()

# Call the function to run the game
if __name__ == "__main__":
    simulate_game()
