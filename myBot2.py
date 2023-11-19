import pygame
import time
import random
import heapq
import math
import copy
import numpy as np
import sys
import csv

pygame.init()
last_total = 1000000
idle = 0
save = 0
# alpha = 0.001
alpha = 0.009
aliens = []
# wait_time = 350
wait_time = 10
k = 1
fps = 60
grid_node_width = 25
grid_node_height = 25
size = 50
simulate = True
total_open = 0
time_step = 0
WIDTH = grid_node_width * size
HEIGHT = grid_node_height * size
screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption('Bot 2  Algo')
img = pygame.image.load('bot.jpg')
img = pygame.transform.scale(img, (grid_node_height - 5, grid_node_width - 5))
img_1 = pygame.image.load('Deoxys.png')
img_1 = pygame.transform.scale(img_1, (grid_node_height - 5, grid_node_width - 5))
img_2 = pygame.image.load('crew.png')
img_2 = pygame.transform.scale(img_2, (grid_node_height - 5, grid_node_width - 5))
crew_alien_grid_size = 7
alien_threshold = 0.05
grid_distances = {}

# creating a closed grid of given dimensions
grid = np.array([[1 for i in range(size)] for j in range(size)])
prob_mat = copy.deepcopy(grid)
prob_mat = [[1 for i in range(size)] for j in range(size)]


# class based representation of a single cell in the graph
class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __repr__(self):
        return f"{self.position} - g: {self.g} h: {self.h} f: {self.f}"

    # defining less than for purposes of heap queue
    def __lt__(self, other):
        return self.f < other.f

    # defining greater than for purposes of heap queue
    def __gt__(self, other):
        return self.f > other.f


# function to create squares while displaying the grid
def createSquare(x, y, color):
    pygame.draw.rect(screen, color, [x, y, grid_node_width, grid_node_height])


# Function to open the grid and create a new configuration
def open_grid(start_x, start_y):
    grid[start_x][start_y] = 0
    global total_open
    while True:
        temp = []
        for i in range(size):
            for j in range(size):
                if grid[i][j] == 1:
                    neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                    count = 0
                    for nx, ny in neighbors:
                        if (nx > -1 and nx < size) and (ny > -1 and ny < size):
                            if grid[nx][ny] == 0:
                                count = count + 1
                if count == 1:
                    temp.append((i, j))
        if len(temp) == 0:
            break
        # randomly picking a cell to be opened from the list of eligible cells
        new_x, new_y = random.choice(temp)
        grid[new_x][new_y] = 0
        # prob_mat[new_x][new_y] = 1
        total_open += 1


# Function to create and return the traversed path
def return_path(current_node):
    path = []
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return path[::-1]  # Return reversed path


# The A* path finding algorithm function
def astar(grid, start, end, aliens):
    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Heapify the open_list and Add the start node
    heapq.heapify(open_list)
    heapq.heappush(open_list, start_node)

    # what squares do we search
    adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0),)

    # Loop until you find the end
    while len(open_list) > 0:
        # Get the current node
        current_node = heapq.heappop(open_list)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            return return_path(current_node)

        # Generate children
        children = []

        for new_position in adjacent_squares:  # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(grid) - 1) or node_position[0] < 0 or node_position[1] > (
                    len(grid[len(grid) - 1]) - 1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if grid[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:
            # Child is on the closed list
            if len([closed_child for closed_child in closed_list if closed_child == child]) > 0:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = abs(child.position[0] - end_node.position[0]) + abs(child.position[1] - end_node.position[1])
            child.f = child.g + child.h

            # Child is already in the open list
            if len([open_node for open_node in open_list if
                    child.position == open_node.position and child.g > open_node.g]) > 0:
                continue

            # Add the child to the open list
            heapq.heappush(open_list, child)

    return None


# Function to display the current grid configurationx
def visualizeGrid(bot_pos, aliens):
    y = 0  # we start at the top of the screen
    for row in grid:
        x = 0  # for every row we start at the left of the screen again
        for item in row:
            if item == 0:
                createSquare(x, y, (255, 255, 255))
            else:
                createSquare(x, y, (0, 0, 0))

            x += grid_node_width  # for ever item/number in that row we move one "step" to the right
        y += grid_node_height  # for every new row we move one "step" downwards
    screen.blit(img, (bot_pos[1] * grid_node_width, bot_pos[0] * grid_node_height))
    screen.blit(img_2, (desired_node[1] * grid_node_width, desired_node[0] * grid_node_height))
    for x, y in aliens:
        screen.blit(img_1, (y * grid_node_width, x * grid_node_height))
    pygame.display.update()


# Function to determine where the aliens would be in the next time step
def new_alien_pos(aliens):
    # choices that the alien can make
    move_set = ((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0))
    new_pos = []
    i = 0
    while len(new_pos) < k:
        current_node = aliens[i]
        new_position = random.choice(move_set)
        # Get node position
        node_position = (current_node[0] + new_position[0], current_node[1] + new_position[1])

        # Make sure within range
        if node_position[0] > (len(grid) - 1) or node_position[0] < 0 or node_position[1] > (
                len(grid[len(grid) - 1]) - 1) or node_position[1] < 0:
            continue

        # Make sure walkable terrain
        if grid[node_position[0]][node_position[1]] != 0:
            continue

        # An alien should not collide with another alien
        if node_position in new_pos:
            continue

        new_pos.append(node_position)
        i += 1
    return new_pos


def detect_aliens(bot_pos, k):
    for column in range(bot_pos[1] - k, bot_pos[1] + k):
        for row in range(bot_pos[0] - k, bot_pos[0] + k):
            if (row, column) in aliens:
                return True
    return False


def move_to_pos(current_position, destination, coordinate):
    move_values = []
    if coordinate == 0:
        if destination[0] > current_position[0]:
            for pos in range(current_position[0] + 1, destination[0] + 1):
                move_values.append(pos)
        else:
            for pos in range(current_position[0] - 1, destination[0] - 1, - 1):
                move_values.append(pos)
    elif coordinate == 1:
        if destination[1] > current_position[1]:
            for pos in range(current_position[1] + 1, destination[1] + 1):
                move_values.append(pos)
        else:
            for pos in range(current_position[1] - 1, destination[1] - 1, - 1):
                move_values.append(pos)
    return move_values


def rand_cords():
    return (random.randint(0, size - 1), random.randint(0, size - 1))


def beep_prob(alpha, d):
    prob = math.exp(-alpha * (d - 1))
    return random.choices([1, 0], weights=[prob, 1 - prob])


def is_valid(node):
    if node[0] < 0 or node[0] >= size:
        return False
    if node[1] < 0 or node[1] >= size:
        return False
    if grid[node[0]][node[1]] == 1:
        return False
    return True

def calc_distances(start_cell):
    queue = []
    visited = []
    queue.append((start_cell, 0))
    grid_distances[start_cell] = {}
    while (len(queue) > 0):
        node, dist = queue.pop(0)
        grid_distances[start_cell][node] = dist
        visited.append(node)

        if is_valid((node[0] - 1, node[1])) and (node[0] - 1, node[1]) not in visited:
            queue.append(((node[0] - 1, node[1]), dist + 1))
        if is_valid((node[0] + 1, node[1])) and (node[0] + 1, node[1]) not in visited:
            queue.append(((node[0] + 1, node[1]), dist + 1))
        if is_valid((node[0], node[1] - 1)) and (node[0], node[1] - 1) not in visited:
            queue.append(((node[0], node[1] - 1), dist + 1))
        if is_valid((node[0], node[1] + 1)) and (node[0], node[1] + 1) not in visited:
            queue.append(((node[0], node[1] + 1), dist + 1))

def update_probs(start_cell):
    total = 0
    for i in range(size):
        for j in range(size):
            if prob_mat[i][j] > -1:
                prob_mat[i][j] = prob_mat[i][j]*(math.exp(-alpha * grid_distances[start_cell][(i, j)]))
                total += prob_mat[i][j]

    for i in range(size):
        for j in range(size):
            if prob_mat[i][j] > -1:
                prob_mat[i][j] = prob_mat[i][j] / total

def update_no_probs(start_cell):
    total = 0
    for i in range(size):
        for j in range(size):
            if prob_mat[i][j] > -1:
                prob_mat[i][j] = prob_mat[i][j]*(1 - math.exp(-alpha * grid_distances[start_cell][(i, j)]))
                total += prob_mat[i][j]

    for i in range(size):
        for j in range(size):
            if prob_mat[i][j] > -1:
                prob_mat[i][j] = prob_mat[i][j] / total

def region_value(start_cell, distance):
    queue = []
    visited = []
    queue.append((start_cell, 0))
    total = 0
    while (len(queue) > 0):
        node, dist = queue.pop(0)
        visited.append(node)
        if prob_mat[node[0]][node[1]] > -1:
            total += prob_mat[node[0]][node[1]]

    if is_valid((node[0] - 1, node[1])) and (node[0] - 1, node[1]) not in visited and dist < distance:
        queue.append(((node[0] - 1, node[1]), dist + 1))
    if is_valid((node[0] + 1, node[1])) and (node[0] + 1, node[1]) not in visited:
        queue.append(((node[0] + 1, node[1]), dist + 1))
    if is_valid((node[0], node[1] - 1)) and (node[0], node[1] - 1) not in visited:
        queue.append(((node[0], node[1] - 1), dist + 1))
    if is_valid((node[0], node[1] + 1)) and (node[0], node[1] + 1) not in visited:
        queue.append(((node[0], node[1] + 1), dist + 1))
    return total

# def update_probs(start_cell):
#     queue = []
#     visited = []
#     queue.append((start_cell, 0))
#     total = 0
#     while (len(queue) > 0):
#         node, dist = queue.pop(0)
#         visited.append(node)
#         if prob_mat[node[0]][node[1]] > -1:
#             prob_mat[node[0]][node[1]] = prob_mat[node[0]][node[1]] * math.exp(-alpha * (dist - 1))
#             total += prob_mat[node[0]][node[1]]

        # if is_valid((node[0] - 1, node[1])) and (node[0] - 1, node[1]) not in visited:
        #     queue.append(((node[0] - 1, node[1]), dist + 1))
        # if is_valid((node[0] + 1, node[1])) and (node[0] + 1, node[1]) not in visited:
        #     queue.append(((node[0] + 1, node[1]), dist + 1))
        # if is_valid((node[0], node[1] - 1)) and (node[0], node[1] - 1) not in visited:
        #     queue.append(((node[0], node[1] - 1), dist + 1))
        # if is_valid((node[0], node[1] + 1)) and (node[0], node[1] + 1) not in visited:
        #     queue.append(((node[0], node[1] + 1), dist + 1))

#     for i in range(size):
#         for j in range(size):
#             if prob_mat[i][j] > -1:
#                 prob_mat[i][j] = prob_mat[i][j] / total
#     # comment this line and check
#     return


# def update_no_probs(start_cell):
#     queue = []
#     visited = []
#     queue.append((start_cell, 0))
#     total = 0
#     while (len(queue) > 0):
#         node, dist = queue.pop(0)
#         visited.append(node)
#         if prob_mat[node[0]][node[1]] > -1:
#             prob_mat[node[0]][node[1]] = prob_mat[node[0]][node[1]] * (1 - math.exp(-alpha * (dist - 1)))
#             total += prob_mat[node[0]][node[1]]

#         if is_valid((node[0] - 1, node[1])) and (node[0] - 1, node[1]) not in visited:
#             queue.append(((node[0] - 1, node[1]), dist + 1))
#         if is_valid((node[0] + 1, node[1])) and (node[0] + 1, node[1]) not in visited:
#             queue.append(((node[0] + 1, node[1]), dist + 1))
#         if is_valid((node[0], node[1] - 1)) and (node[0], node[1] - 1) not in visited:
#             queue.append(((node[0], node[1] - 1), dist + 1))
#         if is_valid((node[0], node[1] + 1)) and (node[0], node[1] + 1) not in visited:
#             queue.append(((node[0], node[1] + 1), dist + 1))

#     for i in range(size):
#         for j in range(size):
#             if prob_mat[i][j] > -1:
#                 prob_mat[i][j] = prob_mat[i][j] / total
#     # comment this line and check
#     return


# oper can be "outsideBorder" "insideBorder" "outsideUpdate" "insideUpdate" "stepOneInside" "stepOneInside"
def update_crew_alien_grid_AlienMoved(grid, crew_alien_grid, x, y, prev, crew_found_inside_detection_grid, blocked_nodes,
                                 movement):
    copy_crew_alien_grid = copy.deepcopy(crew_alien_grid)
    crew_alien_grid = [[0 for i in range(size)] for j in range(size)]

    blocked_nodes_inside_crew_alien = []
    opened_nodes_inside_crew_alien = []
    summation = 0

    if prev is None:
        if crew_found_inside_detection_grid == True:
            oper = "stepOneInside"
        if crew_found_inside_detection_grid == False:
            oper = "stepOneOutside"

    elif crew_found_inside_detection_grid == True:
        if prev == True:
            oper = "insideUpdate"
        if prev == False:
            oper = "insideBorder"

    elif crew_found_inside_detection_grid == False:
        if prev == True:
            oper = "outsideBorder"
        if prev == False:
            oper = "outsideUpdate"

    x_start = x - (crew_alien_grid_size // 2)
    x_end = x - (crew_alien_grid_size // 2) + crew_alien_grid_size - 1
    y_start = y - (crew_alien_grid_size // 2)
    y_end = y - (crew_alien_grid_size // 2) + crew_alien_grid_size - 1

    for i in range(x_start, x_end + 1):
        for j in range(y_start, y_end + 1):
            if i < 0 or j < 0 or i >= size or j >= size:
                continue

            if (i, j) in blocked_nodes:
                blocked_nodes_inside_crew_alien.append((i, j))
            else:
                opened_nodes_inside_crew_alien.append((i, j))

            if oper == "stepOneInside":

                if (i, j) not in blocked_nodes and (i, j) is not (x, y):
                    crew_alien_grid[i][j] += 1
                    summation += crew_alien_grid[i][j]

            if oper == "insideBorder":
                # if (i==x-(crew_alien_grid_size//2) or i== x-(crew_alien_grid_size//2)+ crew_alien_grid_size or j==y-(crew_alien_grid_size//2) or j== y-(crew_alien_grid_size//2)+ crew_alien_grid_size or (i,j) not in blocked_nodes)  :
                #     if
                #         crew_alien_grid[i][j]=1
                #         count_insideb+=1
                if (i, j) in blocked_nodes:
                    continue

                if i == x_start and (i - 1, j) not in blocked_nodes and i > 0:
                    # possible_moves=getPossibleMoves(i-1,j,blocked_nodes)
                    # crew_alien_grid[i][j]+=(copy_crew_alien_grid[i-1][j]/len(possible_moves))
                    # summation+=(copy_crew_alien_grid[i-1][j]/len(possible_moves))
                    summation += copy_crew_alien_grid[i - 1][j]
                    crew_alien_grid[i][j] += copy_crew_alien_grid[i - 1][j]

                if j == y_start and (i, j - 1) not in blocked_nodes and j > 0:
                    # possible_moves=getPossibleMoves(i,j-1,blocked_nodes)
                    # crew_alien_grid[i][j]+=(copy_crew_alien_grid[i][j-1]/len(possible_moves))
                    # summation+=(copy_crew_alien_grid[i][j-1]/len(possible_moves))
                    summation += copy_crew_alien_grid[i][j - 1]
                    crew_alien_grid[i][j] += copy_crew_alien_grid[i][j - 1]

                if i == x_end and (i + 1, j) not in blocked_nodes and i + 1 <= size - 1:
                    # possible_moves=getPossibleMoves(i+1,j,blocked_nodes)
                    # crew_alien_grid[i][j]+=(copy_crew_alien_grid[i+1][j]/len(possible_moves))
                    # summation+=(copy_crew_alien_grid[i+1][j]/len(possible_moves))
                    summation += copy_crew_alien_grid[i + 1][j]
                    crew_alien_grid[i][j] += copy_crew_alien_grid[i + 1][j]

                if j == y_end and (i, j + 1) not in blocked_nodes and j + 1 <= size - 1:
                    # possible_moves=getPossibleMoves(i,j+1,blocked_nodes)
                    # crew_alien_grid[i][j]+=(copy_crew_alien_grid[i][j+1]/len(possible_moves))
                    # summation+=(copy_crew_alien_grid[i][j+1]/len(possible_moves))
                    summation += copy_crew_alien_grid[i][j + 1]
                    crew_alien_grid[i][j] += copy_crew_alien_grid[i][j + 1]

                if movement == (1, 0) and i == x_end and j in range(y_start, y_end + 1):
                    possible_moves = getPossibleMoves(i, j, blocked_nodes)
                    moveInsideBoundary = []
                    summation += copy_crew_alien_grid[i][j]
                    for possible_move in possible_moves:
                        if possible_move[0] < x_start or possible_move[1] < y_start or possible_move[0] > x_end or \
                                possible_move[1] > y_end:
                            continue

                        moveInsideBoundary.append((possible_move))

                    for mov in moveInsideBoundary:
                        crew_alien_grid[mov[0]][mov[1]] += (copy_crew_alien_grid[i][j] / len(moveInsideBoundary))

                if movement == (-1, 0) and i == x_start and j in range(y_start, y_end + 1):
                    possible_moves = getPossibleMoves(i, j, blocked_nodes)
                    moveInsideBoundary = []
                    summation += copy_crew_alien_grid[i][j]
                    for possible_move in possible_moves:
                        if possible_move[0] < x_start or possible_move[1] < y_start or possible_move[0] > x_end or \
                                possible_move[1] > y_end:
                            continue

                        moveInsideBoundary.append((possible_move))

                    for mov in moveInsideBoundary:
                        crew_alien_grid[mov[0]][mov[1]] += (copy_crew_alien_grid[i][j] / len(moveInsideBoundary))

                if movement == (0, 1) and j == y_end and i in range(x_start, x_end + 1):
                    possible_moves = getPossibleMoves(i, j, blocked_nodes)
                    moveInsideBoundary = []
                    summation += copy_crew_alien_grid[i][j]
                    for possible_move in possible_moves:
                        if possible_move[0] < x_start or possible_move[1] < y_start or possible_move[0] > x_end or \
                                possible_move[1] > y_end:
                            continue

                        moveInsideBoundary.append((possible_move))

                    for mov in moveInsideBoundary:
                        crew_alien_grid[mov[0]][mov[1]] += (copy_crew_alien_grid[i][j] / len(moveInsideBoundary))

                if movement == (0, -1) and j == y_start and i in range(x_start, x_end + 1):
                    possible_moves = getPossibleMoves(i, j, blocked_nodes)
                    moveInsideBoundary = []
                    summation += copy_crew_alien_grid[i][j]
                    for possible_move in possible_moves:
                        if possible_move[0] < x_start or possible_move[1] < y_start or possible_move[0] > x_end or \
                                possible_move[1] > y_end:
                            continue

                        moveInsideBoundary.append((possible_move))

                    for mov in moveInsideBoundary:
                        crew_alien_grid[mov[0]][mov[1]] += (copy_crew_alien_grid[i][j] / len(moveInsideBoundary))



            elif oper == "outsideBorder":
                if (i, j) in blocked_nodes:
                    continue

                elif i == x_start and j == y_start:
                    possible_moves = []
                    if (i - 1, j) not in blocked_nodes and i - 1 >= 0:
                        possible_moves.append((i - 1, j))

                    if (i, j - 1) not in blocked_nodes and j - 1 >= 0:
                        possible_moves.append((i, j - 1))

                    if len(possible_moves) == 0:
                        continue
                    summation += copy_crew_alien_grid[i][j]
                    for possible_move in possible_moves:
                        crew_alien_grid[possible_move[0]][possible_move[1]] += copy_crew_alien_grid[i][j] / len(
                            possible_moves)

                elif i == x_start and j == y_end:
                    possible_moves = []
                    if (i - 1, j) not in blocked_nodes and i - 1 >= 0:
                        possible_moves.append((i - 1, j))

                    if (i, j + 1) not in blocked_nodes and j + 1 <= size - 1:
                        possible_moves.append((i, j + 1))

                    if len(possible_moves) == 0:
                        continue
                    summation += copy_crew_alien_grid[i][j]
                    for possible_move in possible_moves:
                        crew_alien_grid[possible_move[0]][possible_move[1]] += copy_crew_alien_grid[i][j] / len(
                            possible_moves)

                elif i == x_end and j == y_start:
                    possible_moves = []
                    if (i + 1, j) not in blocked_nodes and i + 1 <= size - 1:
                        possible_moves.append((i + 1, j))

                    if (i, j - 1) not in blocked_nodes and j - 1 >= 0:
                        possible_moves.append((i, j - 1))

                    if len(possible_moves) == 0:
                        continue
                    summation += copy_crew_alien_grid[i][j]
                    for possible_move in possible_moves:
                        crew_alien_grid[possible_move[0]][possible_move[1]] += copy_crew_alien_grid[i][j] / len(
                            possible_moves)

                elif i == x_start and j == y_start:
                    possible_moves = []
                    if (i + 1, j) not in blocked_nodes and i + 1 <= size - 1:
                        possible_moves.append((i + 1, j))

                    if (i, j + 1) not in blocked_nodes and j + 1 <= size - 1:
                        possible_moves.append((i, j + 1))

                    if len(possible_moves) == 0:
                        continue
                    summation += copy_crew_alien_grid[i][j]
                    for possible_move in possible_moves:
                        crew_alien_grid[possible_move[0]][possible_move[1]] += copy_crew_alien_grid[i][j] / len(
                            possible_moves)


                elif i == x_start and (i - 1, j) not in blocked_nodes and i > 0:
                    crew_alien_grid[i - 1][j] += copy_crew_alien_grid[i][j]
                    summation += copy_crew_alien_grid[i][j]

                elif j == y_start and (i, j - 1) not in blocked_nodes and j > 0:
                    crew_alien_grid[i][j - 1] += copy_crew_alien_grid[i][j]
                    summation += copy_crew_alien_grid[i][j]

                elif i == x_end and (i + 1, j) not in blocked_nodes and i != size - 1:
                    crew_alien_grid[i + 1][j] += copy_crew_alien_grid[i][j]
                    summation += copy_crew_alien_grid[i][j]

                elif j == y_end and (i, j + 1) not in blocked_nodes and j != size - 1:
                    crew_alien_grid[i][j + 1] += copy_crew_alien_grid[i][j]
                    summation += copy_crew_alien_grid[i][j]


            elif oper == "insideUpdate":
                # how many possible values
                if (i, j) not in blocked_nodes and (x, y) is not (i, j):
                    move_set = ((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0))
                    possible_movements = []
                    summation += copy_crew_alien_grid[i][j]
                    for move in move_set:
                        adj_i = i + move[0]
                        adj_j = j + move[1]
                        if adj_i < 0 or adj_j < 0 or adj_i >= size or adj_j >= size or (adj_i, adj_j) is (x, y):
                            continue

                        if (adj_i, adj_j) not in blocked_nodes and not (
                                adj_i < x_start or adj_j < y_start or adj_i > x_end or adj_j > y_end):
                            possible_movements.append((adj_i, adj_j))

                    for possible_movement in possible_movements:
                        crew_alien_grid[possible_movement[0]][possible_movement[1]] += copy_crew_alien_grid[i][j] / len(
                            possible_movements)

    if oper == "outsideBorder":

        if movement == (1, 0):
            if x_start - 1 >= 0:
                for j in range(y_start, y_end + 1):
                    if j >= 0 and j <= size - 1:
                        possible_moves = getPossibleMoves(x_start - 1, j,
                                                          blocked_nodes + opened_nodes_inside_crew_alien)
                        summation += copy_crew_alien_grid[x_start - 1][j]
                        for poss_move in possible_moves:
                            crew_alien_grid[poss_move[0]][poss_move[1]] += (
                                        copy_crew_alien_grid[x_start - 1][j] / len(possible_moves))

        if movement == (-1, 0):
            if x_end + 1 <= size - 1:
                for j in range(y_start, y_end + 1):
                    if j >= 0 and j <= size - 1:
                        possible_moves = getPossibleMoves(x_end + 1, j, blocked_nodes + opened_nodes_inside_crew_alien)
                        summation += copy_crew_alien_grid[x_end + 1][j]
                        for poss_move in possible_moves:
                            crew_alien_grid[poss_move[0]][poss_move[1]] += (
                                        copy_crew_alien_grid[x_end + 1][j] / len(possible_moves))

        if movement == (0, 1):
            if y_start - 1 >= 0:
                for i in range(x_start, x_end + 1):
                    if i >= 0 and i <= size - 1:
                        possible_moves = getPossibleMoves(i, y_start - 1,
                                                          blocked_nodes + opened_nodes_inside_crew_alien)
                        summation += copy_crew_alien_grid[i][y_start - 1]
                        for poss_move in possible_moves:
                            crew_alien_grid[poss_move[0]][poss_move[1]] += (
                                        copy_crew_alien_grid[i][y_start - 1] / len(possible_moves))

        if movement == (0, -1):
            if y_end + 1 <= size - 1:
                for i in range(x_start, x_end + 1):
                    if i >= 0 and i <= size - 1:
                        possible_moves = getPossibleMoves(i, y_end + 1, blocked_nodes + opened_nodes_inside_crew_alien)
                        summation += copy_crew_alien_grid[i][y_end + 1]
                        for poss_move in possible_moves:
                            crew_alien_grid[poss_move[0]][poss_move[1]] += (
                                        copy_crew_alien_grid[i][y_end + 1] / len(possible_moves))

    if oper == "outsideUpdate":

        for i in range(size):
            for j in range(size):
                if (i, j) not in blocked_nodes:
                    move_set = ((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0))
                    possible_movements = []
                    summation += copy_crew_alien_grid[i][j]
                    for move in move_set:
                        adj_i = i + move[0]
                        adj_j = j + move[1]
                        if adj_i < 0 or adj_j < 0 or adj_i >= size or adj_j >= size:
                            continue

                        if (adj_i, adj_j) not in blocked_nodes and (adj_i, adj_j) not in opened_nodes_inside_crew_alien:
                            possible_movements.append((adj_i, adj_j))

                    for possible_movement in possible_movements:
                        crew_alien_grid[possible_movement[0]][possible_movement[1]] += copy_crew_alien_grid[i][j] / len(
                            possible_movements)

    if oper == "insideUpdate":
        if movement == (1, 0):
            if x_start - 1 >= 0:
                for j in range(y_start, y_end + 1):
                    if j >= 0 and j <= size - 1:
                        summation += copy_crew_alien_grid[x_start - 1][j]
                        crew_alien_grid[x_start][j] += copy_crew_alien_grid[x_start - 1][j]

        if movement == (-1, 0):
            if x_end + 1 <= size - 1:
                for j in range(y_start, y_end + 1):
                    if j >= 0 and j <= size - 1:
                        crew_alien_grid[x_end][j] += copy_crew_alien_grid[x_end + 1][j]
                        summation += copy_crew_alien_grid[x_end + 1][j]

        if movement == (0, 1):
            if y_start - 1 >= 0:
                for i in range(x_start, x_end + 1):
                    if i >= 0 and i <= size - 1:
                        crew_alien_grid[i][y_start] += copy_crew_alien_grid[i][y_start - 1]
                        summation += copy_crew_alien_grid[i][y_start - 1]

        if movement == (0, -1):
            if y_end + 1 <= size - 1:
                for i in range(x_start, x_end + 1):
                    if i >= 0 and i <= size - 1:
                        crew_alien_grid[i][y_end] += copy_crew_alien_grid[i][y_end + 1]
                        summation += copy_crew_alien_grid[i][y_end + 1]

    if oper == "stepOneOutside":
        for i in range(size):
            for j in range(size):
                if (i, j) not in blocked_nodes and (i, j) not in opened_nodes_inside_crew_alien:
                    crew_alien_grid[i][j] = 1
                    summation += crew_alien_grid[i][j]

    
    if summation<=0:
        return copy_crew_alien_grid
                    ##Probalizing the values
    for i in range(size):
        for j in range(size):
            crew_alien_grid[i][j] = crew_alien_grid[i][j] / summation

    return crew_alien_grid


def update_crew_alien_grid_CrewMoved(grid, crew_alien_grid, x, y, prev, crew_found_inside_detection_grid, blocked_nodes,
                                 movement):
    copy_crew_alien_grid = copy.deepcopy(crew_alien_grid)
    crew_alien_grid = [[0 for i in range(size)] for j in range(size)]

    blocked_nodes_inside_crew_alien = []
    opened_nodes_inside_crew_alien = []
    summation = 0

    if prev is None:
        if crew_found_inside_detection_grid == True:
            oper = "stepOneInside"
        if crew_found_inside_detection_grid == False:
            oper = "stepOneOutside"

    elif crew_found_inside_detection_grid == True:
        oper="inside"
        # if prev == True:
        #     oper = "insideUpdate"
        # if prev == False:
        #     oper = "insideBorder"

    elif crew_found_inside_detection_grid == False:
        oper="outside"
        # if prev == True:
        #     oper = "outsideBorder"
        # if prev == False:
        #     oper = "outsideUpdate"

    x_start = x - (crew_alien_grid_size // 2)
    x_end = x - (crew_alien_grid_size // 2) + crew_alien_grid_size - 1
    y_start = y - (crew_alien_grid_size // 2)
    y_end = y - (crew_alien_grid_size // 2) + crew_alien_grid_size - 1

    for i in range(x_start, x_end + 1):
        for j in range(y_start, y_end + 1):
            if i < 0 or j < 0 or i >= size or j >= size:
                continue

            if (i, j) in blocked_nodes:
                blocked_nodes_inside_crew_alien.append((i, j))
            else:
                opened_nodes_inside_crew_alien.append((i, j))

            if oper == "stepOneInside":

                if (i, j) not in blocked_nodes and (i, j) is not (x, y):
                    crew_alien_grid[i][j] += 1
                    summation += crew_alien_grid[i][j]

            elif oper == "inside"  :
                # how many possible values
                if (i, j) not in blocked_nodes :
                    crew_alien_grid[i][j]+=copy_crew_alien_grid[i][j]
                    
                    summation += copy_crew_alien_grid[i][j]
                    

    if oper == "outside":

        for i in range(size):
            for j in range(size):
                if (i, j) not in blocked_nodes and (i,j) not in opened_nodes_inside_crew_alien:
                    crew_alien_grid[i][j]+=copy_crew_alien_grid[i][j]
                    summation += copy_crew_alien_grid[i][j]
                    
                    



    if oper == "stepOneOutside":
        for i in range(size):
            for j in range(size):
                if (i, j) not in blocked_nodes and (i, j) not in opened_nodes_inside_crew_alien:
                    crew_alien_grid[i][j] = 1
                    summation += crew_alien_grid[i][j]

    if summation<=0:
        return copy_crew_alien_grid

                    ##Probalizing the values
    for i in range(size):
        for j in range(size):
            crew_alien_grid[i][j] = crew_alien_grid[i][j] / summation

    return crew_alien_grid



def getPossibleMoves(p, q, blocked_nodes):
    move_set = ((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0))
    possibles = []
    for move in move_set:
        p_adj = p + move[0]
        q_adj = q + move[1]
        if p_adj < 0 or q_adj < 0 or p_adj >= size or q_adj >= size:
            continue
        if (p_adj, q_adj) not in blocked_nodes:
            possibles.append((p_adj, q_adj))

    return possibles


def find_crew_found_inside_detection_grid(grid, aliens, x, y):
    x_start = x - (crew_alien_grid_size // 2)
    x_end = x - (crew_alien_grid_size // 2) + crew_alien_grid_size - 1
    y_start = y - (crew_alien_grid_size // 2)
    y_end = y - (crew_alien_grid_size // 2) + crew_alien_grid_size - 1

    for i in range(x_start, x_end + 1):
        for j in range(y_start, y_end + 1):
            if (i, j) in aliens:
                return True
    return False



def getBestPossibleNeighbour(node, blocked_nodes, crew_alien_grid):
    move_set = ((0, -1), (0, 1), (-1, 0), (1, 0))
    possibles = None
    probMin = float("inf")
    for move in move_set:
        p_adj = node[0] + move[0]
        q_adj = node[1] + move[1]
        if p_adj < 0 or q_adj < 0 or p_adj >= size or q_adj >= size:
            continue
        if (p_adj, q_adj) not in blocked_nodes:
            if crew_alien_grid[p_adj][q_adj] < probMin and crew_alien_grid[p_adj][q_adj] <= alien_threshold :
                probMin = crew_alien_grid[p_adj][q_adj]
                possibles = (p_adj, q_adj)
    if possibles is None:
        possibles = node
    return possibles


def probOfComingToTheBotPosition(node, blocked_nodes, crew_alien_grid):
    move_set = ((0, -1), (0, 1), (-1, 0), (1, 0))
    summing = 0
    for move in move_set:
        p_adj = node[0] + move[0]
        q_adj = node[1] + move[1]
        if p_adj < 0 or q_adj < 0 or p_adj >= size or q_adj >= size:
            continue
        if (p_adj, q_adj) not in blocked_nodes:
            length = len(getPossibleMoves(p_adj, q_adj, blocked_nodes))
            if length > 1:
                summing += (crew_alien_grid[p_adj][q_adj] / length)
    print(summing)
    return summing



# MIN

open_node = rand_cords()
open_grid(open_node[0], open_node[1])

crew_alien_grid = [[0 for i in range(size)] for j in range(size)]
prev = None

# generating a li     st of all the cells that are blocked
blocked_nodes = []
for i in range(size):
    for j in range(size):
        if grid[i][j] == 1:
            blocked_nodes.append((i, j))
            prob_mat[i][j] = -1

initial_node = rand_cords()

while initial_node in blocked_nodes:
    initial_node = rand_cords()

while len(aliens) < k:
    new_node = rand_cords()
    if new_node not in blocked_nodes and new_node != initial_node and new_node not in aliens:
        aliens.append(new_node)

desired_node = rand_cords()
while desired_node in blocked_nodes or desired_node == initial_node or desired_node in aliens:
    desired_node = rand_cords()



total_open = total_open - 1

for i in range(size):
    for j in range(size):
        if prob_mat[i][j] > -1:
            prob_mat[i][j] = float(prob_mat[i][j]) / float(total_open)
print("The value of the grid distances dictionary are ", grid_distances)
prob_mat[initial_node[0]][initial_node[1]] = -1

no_of_moves=0
moves = 0
prev_node = initial_node
f=open('bot2AnalysisFinal.csv','a')
writer=csv.writer(f)
ans = 0
idle_time=0
dist = None
while simulate:
    flag=0
    while initial_node != desired_node:
        no_of_moves+=1
        if initial_node == desired_node:
            writer.writerow([alpha,crew_alien_grid_size,alien_threshold,no_of_moves,idle_time,"crew Found",dist])
            f.close()
            sys.exit(1)
        print("Inside the while loop")
        visualizeGrid(initial_node, aliens)
        #time.sleep(0.2)
        crew_found_inside_detection_grid = find_crew_found_inside_detection_grid(grid, aliens, initial_node[0],
                                                                                 initial_node[1])
        crew_alien_grid = update_crew_alien_grid_CrewMoved(grid, crew_alien_grid, initial_node[0], initial_node[1], prev,
                                                       crew_found_inside_detection_grid, blocked_nodes,
                                                       (initial_node[0] - prev_node[0], initial_node[1] - prev_node[1]))
        prev = crew_found_inside_detection_grid
        prev_node=initial_node
        
        
        
        d = len(astar(grid, initial_node, desired_node, aliens))
        ans = beep_prob(alpha, d)[0]
        if initial_node not in grid_distances:
            calc_distances(initial_node)
        print("Updating the probs")
        if ans == 1:
            update_probs(initial_node)
        elif ans == 0:
            update_no_probs(initial_node)
        # check for alien detection
        aliens = new_alien_pos(aliens)
        if initial_node in aliens:
            # Alien enters bot occupied cell
            writer.writerow([alpha,crew_alien_grid_size,alien_threshold,no_of_moves,idle_time,"alien Found",dist])
            f.close()
            sys.exit(1)
        # Update alien probabitlies
        crew_found_inside_detection_grid = find_crew_found_inside_detection_grid(grid, aliens, initial_node[0],
                                                                                 initial_node[1])
        crew_alien_grid = update_crew_alien_grid_AlienMoved(grid, crew_alien_grid, initial_node[0], initial_node[1], prev,
                                                       crew_found_inside_detection_grid, blocked_nodes,
                                                       (initial_node[0] - prev_node[0], initial_node[1] - prev_node[1]))
        #movement will always be (0,0) since prev node has been set to initial node
        prev = crew_found_inside_detection_grid
        prev_node=initial_node
        
        
        if probOfComingToTheBotPosition(initial_node, blocked_nodes, crew_alien_grid) > alien_threshold:
            initial_node = getBestPossibleNeighbour(initial_node, blocked_nodes, crew_alien_grid)
            prev_node = initial_node
            idle_time+=1
            #aliens = new_alien_pos(aliens)
            flag = 1
        if flag == 1:
            break
        
        max_value = -1
        max_nodes = []
        # for i in range(size):
        #     for j in range(size):
        #         if prob_mat[i][j] > max_prob:
        #             max_prob = prob_mat[i][j]
        #             max_nodes = [(i, j)]
        #         elif prob_mat[i][j] == max_prob: 
        #             max_nodes.append((i, j))
        dist = 7
        for i in range(size):
            for j in range(size):
                if region_value((i, j), dist) > max_value:
                    max_value = region_value((i, j), dist)
                    max_nodes = [(i, j)]
                elif region_value((i, j), dist):
                    max_nodes.append((i, j))
        print("max_value = ", max_value, "max_nodes = ", max_nodes)
        move_to = random.choice(max_nodes)
        print("The move to value is ", move_to)
        path = astar(grid, initial_node, move_to, aliens)
        initial_node = path[1]
        print("moving to ", initial_node)
        prob_mat[initial_node[0]][initial_node[1]] = -1
        if initial_node in aliens:
            # bot entered alien cells
            writer.writerow([alpha,crew_alien_grid_size,alien_threshold,no_of_moves,idle_time,"alien Found",dist])
            f.close()
            sys.exit(1)
    if  initial_node == desired_node  :
        writer.writerow([alpha,crew_alien_grid_size,alien_threshold,no_of_moves,idle_time,"crew Found",dist])
        f.close()
        sys.exit(1)
pygame.quit()