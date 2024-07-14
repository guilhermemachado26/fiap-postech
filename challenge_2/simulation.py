import pygame

pygame.init()
screen = pygame.display.set_mode((700,700))
pygame.display.set_caption("Game")

speed = 50
x = 50
y = 50

map = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,1,0,1,0,0,0,0,0,1],
    [1,0,1,0,0,0,1,0,1,1,1,0,1],
    [1,0,0,0,1,1,1,0,0,0,0,0,1],
    [1,0,1,0,0,0,0,0,1,1,1,0,1],
    [1,0,1,0,1,1,1,0,1,0,0,0,1],
    [1,0,1,0,1,0,0,0,1,1,1,0,1],
    [1,0,1,0,1,1,1,0,1,0,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,1,2,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1]
]

def game_map():
    global goal
    goal_surface = pygame.Surface((80, 80), pygame.SRCALPHA)
    goal = pygame.draw.rect(goal_surface, (255, 255, 0), (0, 0, 50, 50))
    
    global wall
    wall_surface = pygame.Surface((100, 100), pygame.SRCALPHA)
    wall = pygame.draw.rect(wall_surface, (0, 0, 255), (0, 0, 50, 50)) 

    global path
    path_surface = pygame.Surface((80, 80), pygame.SRCALPHA)
    path = pygame.draw.rect(path_surface, (255, 255, 255), (0, 0, 50, 50)) 

    mapping = {
        0: path_surface,
        1: wall_surface,
        2: goal_surface,
    }

    for y, row in enumerate(map):
        for x, cell in enumerate(row):
            image = mapping[cell]
            screen.blit(image, [x*50, y*50])

    pygame.display.update() 


def player():
    player = pygame.draw.rect(screen, (255,0,0), (x, y, 50, 50))   

loop = True

while loop:
    pygame.time.delay(100)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            loop = False

    # Get the key pressed
    keys = pygame.key.get_pressed()
    
    pos = x, y
    if keys[pygame.K_LEFT]:
        x -= speed
    if keys[pygame.K_RIGHT]:
        x += speed
    if keys[pygame.K_UP]:
        y -= speed
    if keys[pygame.K_DOWN]:
        y += speed

    row = y // 50
    column = x // 50
    if map[row][column] == 1:
        x, y = pos

    if map[row][column] == 2:
        print('You Win')

    screen.fill((255,255,255))
    game_map()
    player()
    pygame.display.update()

pygame.quit()   