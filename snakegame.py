from Genetic_Algorithm import *
from Snake_Game import *
sol_per_pop = 50
num_weights = n_x*n_h + n_h*n_h2 + n_h2*n_y
pop_size = (sol_per_pop,num_weights)
new_population = np.random.choice(np.arange(-1,1,step=0.01),size=pop_size,replace=True)
num_generations = 100
num_parents_mating = 12
for generation in range(num_generations):
    print('##############        GENERATION ' + str(generation)+ '  ###############' )
    fitness = cal_pop_fitness(new_population)
    print('#######  fittest chromosome in gneneration ' + str(generation) +' is having fitness value:  ', np.max(fitness))
    parents = select_mating_pool(new_population, fitness, num_parents_mating)
    offspring_crossover = crossover(parents, offspring_size=(pop_size[0] - parents.shape[0], num_weights))
    offspring_mutation = mutation(offspring_crossover)
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation
from random import choice, randint
def cal_pop_fitness(pop):
    fitness = []
    for i in range(pop.shape[0]):
        fit = run_game_with_ML(display,clock,pop[i])
@@ -20,13 +21,15 @@ def select_mating_pool(pop, fitness, num_parents):
    return parents
def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]):
    for k in range(offspring_size[0]): 
        while True:
            parent1_idx = random.randint(0, parents.shape[0] - 1)
            parent2_idx = random.randint(0, parents.shape[0] - 1)
            # produce offspring from two parents if they are different
            if parent1_idx != parent2_idx:
                for j in range(offspring_size[1]):
                    if random.uniform(0, 1) < 0.5:
def mutation(offspring_crossover):
    for idx in range(offspring_crossover.shape[0]):
        for _ in range(25):
            i = randint(0,offspring_crossover.shape[1]-1)
        random_value = np.random.choice(np.arange(-1,1,step=0.001),size=(1),replace=False)
        offspring_crossover[idx, i] = offspring_crossover[idx, i] + random_value
    return offspring_crossover 
    return offspring_crossover
from Snake_Game import *
from Feed_Forward_Neural_Network import *
def run_game_with_ML(display, clock, weights):
    max_score = 0
    avg_score = 0
    test_games = 1
    score1 = 0
    steps_per_game = 2500
    score2 = 0
    for _ in range(test_games):
        snake_start, snake_position, apple_position, score = starting_positions()
        count_same_direction = 0
        prev_direction = 0
        for _ in range(steps_per_game):
            current_direction_vector, is_front_blocked, is_left_blocked, is_right_blocked = blocked_directions(
                snake_position)
            angle, snake_direction_vector, apple_direction_vector_normalized, snake_direction_vector_normalized = angle_with_apple(
                snake_position, apple_position)
            predictions = []
            predicted_direction = np.argmax(np.array(forward_propagation(np.array(
                [is_left_blocked, is_front_blocked, is_right_blocked, apple_direction_vector_normalized[0],
                 snake_direction_vector_normalized[0], apple_direction_vector_normalized[1],
                 snake_direction_vector_normalized[1]]).reshape(-1, 7), weights))) - 1
            if predicted_direction == prev_direction:
                count_same_direction += 1
            else:
                count_same_direction = 0
                prev_direction = predicted_direction
            new_direction = np.array(snake_position[0]) - np.array(snake_position[1])
            if predicted_direction == -1:
                new_direction = np.array([new_direction[1], -new_direction[0]])
            if predicted_direction == 1:
                new_direction = np.array([-new_direction[1], new_direction[0]])
            button_direction = generate_button_direction(new_direction)
            next_step = snake_position[0] + current_direction_vector
            if collision_with_boundaries(snake_position[0]) == 1 or collision_with_self(next_step.tolist(), snake_position) == 1:
                score1 += -150
                break
            else:
                score1 += 0
            snake_position, apple_position, score = play_game(snake_start, snake_position, apple_position, button_direction, score, display, clock)
            if score > max_score:
                max_score = score
            if count_same_direction > 8 and predicted_direction != 0:
                score2 -= 1
            else:
                score2 += 2
    return score1 + score2 + max_score * 5000
import random
import random
import time
import math
from tqdm import tqdm
import numpy as np
import pygame
def display_snake(snake_position, display):
    for position in snake_position:
        pygame.draw.rect(display, (255, 0, 0), pygame.Rect(position[0], position[1], 10, 10))
def display_apple(apple_position, display):
    pygame.draw.rect(display, (0, 255, 0), pygame.Rect(apple_position[0], apple_position[1], 10, 10))
def starting_positions():
    snake_start = [100, 100]
    snake_position = [[100, 100], [90, 100], [80, 100]]
    apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
    score = 0
    return snake_start, snake_position, apple_position, score
def apple_distance_from_snake(apple_position, snake_position):
    return np.linalg.norm(np.array(apple_position) - np.array(snake_position[0]))
def generate_snake(snake_start, snake_position, apple_position, button_direction, score):
    if button_direction == 1:
        snake_start[0] += 10
    elif button_direction == 0:
        snake_start[0] -= 10
    elif button_direction == 2:
        snake_start[1] += 10
    else:
        snake_start[1] -= 10

    if snake_start == apple_position:
        apple_position, score = collision_with_apple(apple_position, score)
        snake_position.insert(0, list(snake_start))
    else:
        snake_position.insert(0, list(snake_start))
        snake_position.pop()
    return snake_position, apple_position, score
def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
    score += 1
    return apple_position, score
def collision_with_boundaries(snake_start):
    if snake_start[0] >= 500 or snake_start[0] < 0 or snake_start[1] >= 500 or snake_start[1] < 0:
        return 1
    else:
        return 0
def collision_with_self(snake_start, snake_position):
    # snake_start = snake_position[0]
    if snake_start in snake_position[1:]:
        return 1
    else:
        return 0
def blocked_directions(snake_position):
    current_direction_vector = np.array(snake_position[0]) - np.array(snake_position[1])
    left_direction_vector = np.array([current_direction_vector[1], -current_direction_vector[0]])
    right_direction_vector = np.array([-current_direction_vector[1], current_direction_vector[0]])
    is_front_blocked = is_direction_blocked(snake_position, current_direction_vector)
    is_left_blocked = is_direction_blocked(snake_position, left_direction_vector)
    is_right_blocked = is_direction_blocked(snake_position, right_direction_vector)
    return current_direction_vector, is_front_blocked, is_left_blocked, is_right_blocked
def is_direction_blocked(snake_position, current_direction_vector):
    next_step = snake_position[0] + current_direction_vector
    snake_start = snake_position[0]
    if collision_with_boundaries(next_step) == 1 or collision_with_self(next_step.tolist(), snake_position) == 1:
        return 1
    else:
        return 0
def generate_random_direction(snake_position, angle_with_apple):
    direction = 0
    if angle_with_apple > 0:
        direction = 1
    elif angle_with_apple < 0:
        direction = -1
    else:
        direction = 0
    return direction_vector(snake_position, angle_with_apple, direction)
def direction_vector(snake_position, angle_with_apple, direction):
    current_direction_vector = np.array(snake_position[0]) - np.array(snake_position[1])
    left_direction_vector = np.array([current_direction_vector[1], -current_direction_vector[0]])
    right_direction_vector = np.array([-current_direction_vector[1], current_direction_vector[0]])

    new_direction = current_direction_vector
    if direction == -1:
        new_direction = left_direction_vector
    if direction == 1:
        new_direction = right_direction_vector
    button_direction = generate_button_direction(new_direction)
    return direction, button_direction
def generate_button_direction(new_direction):
    button_direction = 0
    if new_direction.tolist() == [10, 0]:
        button_direction = 1
    elif new_direction.tolist() == [-10, 0]:
        button_direction = 0
    elif new_direction.tolist() == [0, 10]:
        button_direction = 2
    else:
        button_direction = 3
    return button_direction
def angle_with_apple(snake_position, apple_position):
    apple_direction_vector = np.array(apple_position) - np.array(snake_position[0])
    snake_direction_vector = np.array(snake_position[0]) - np.array(snake_position[1])
    norm_of_apple_direction_vector = np.linalg.norm(apple_direction_vector)
    norm_of_snake_direction_vector = np.linalg.norm(snake_direction_vector)
    if norm_of_apple_direction_vector == 0:
        norm_of_apple_direction_vector = 10
    if norm_of_snake_direction_vector == 0:
        norm_of_snake_direction_vector = 10
    apple_direction_vector_normalized = apple_direction_vector / norm_of_apple_direction_vector
    snake_direction_vector_normalized = snake_direction_vector / norm_of_snake_direction_vector
    angle = math.atan2(
        apple_direction_vector_normalized[1] * snake_direction_vector_normalized[0] - apple_direction_vector_normalized[
            0] * snake_direction_vector_normalized[1],
        apple_direction_vector_normalized[1] * snake_direction_vector_normalized[1] + apple_direction_vector_normalized[
            0] * snake_direction_vector_normalized[0]) / math.pi
    return angle, snake_direction_vector, apple_direction_vector_normalized, snake_direction_vector_normalized
def play_game(snake_start, snake_position, apple_position, button_direction, score, display, clock):
    crashed = False
    while crashed is not True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
        display.fill((255, 255, 255))
        display_apple(apple_position, display)
        display_snake(snake_position, display)
        snake_position, apple_position, score = generate_snake(snake_start, snake_position, apple_position, button_direction, score)
        pygame.display.set_caption("SCORE: " + str(score))
        pygame.display.update()
        clock.tick(50000)
        return snake_position, apple_position, score
display_width = 500
display_height = 500
green = (0,255,0)
red = (255,0,0)
black = (0,0,0)
white = (255,255,255)
pygame.init()
display=pygame.display.set_mode((display_width,display_height))
clock=pygame.time.Clock()
