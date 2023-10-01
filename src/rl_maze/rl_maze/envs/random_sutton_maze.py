from dis import dis
import numpy as np

def random_sutton_maze(size, difficulty, seed = 0):
    """
    The algorithm is as follows:
    1. Create 
    """
    np.random.seed(seed)
    x = np.zeros((size, size), dtype=np.uint8)
    x[0, :] = 1
    x[-1, :] = 1
    x[:, 0] = 1
    x[:, -1] = 1
    x[1, 1] = 0
    x[-2, -2] = 3
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            if i == 1 and j == 1:
                continue
            if i == size - 2 and j == size - 2:
                continue
            if np.random.rand() < difficulty:
                x[i, j] = 2
    return x

def random_sutton_maze2(size, difficulty, seed = 0):
    """
    The algorithm is as follows:
    1. Create a empty environment with boundaries as 1
    2. Fix the start and goal states
    3. Randomly place fixed number of obstacles
    4. Check the minimum number of steps from start to goal
    5. Based on the difficulty level, there should be fixed number of obstacles and number of steps within which
        agent has to solve the environment
    """
    np.random.seed(seed)
    x = np.zeros((size, size), dtype=np.uint8)
    x[0, :] = 1
    x[-1, :] = 1
    x[:, 0] = 1
    x[:, -1] = 1
    x[1, 1] = 0
    x[-2, -2] = 3
    if difficulty == 1:
        num_obstacles = 7
    elif difficulty == 2:
        num_obstacles = 10
    elif difficulty == 3:
        num_obstacles = 13
    elif difficulty == 4:
        num_obstacles = 14
    else:
        raise ValueError("Difficulty level should be between 1 and 3")
    for i in range(num_obstacles):
        while True:
            x1 = np.random.randint(1, size - 1)
            y1 = np.random.randint(1, size - 1)
            # check if the randomly selected cell is not the start or goal state
            if x1 == 1 and y1 == 1:
                continue
            elif x1 == size - 2 and y1 == size - 2:
                continue
            elif x[x1, y1] == 0:
                x[x1, y1] = 2
                break
    return x

def check_solvability_and_steps_maze2(maze, difficulty):
    if difficulty == 1:
        min_steps = 8
    elif difficulty == 2:
        min_steps = 11
    elif difficulty == 3:
        min_steps = 14
    elif difficulty == 4:
        min_steps = 15
    # check if maze is solvable and the minimum number of steps to reach the goal
    if not check_maze(maze):
        return False
    size = maze.shape[0]
    start = (1, 1)
    goal = (size - 2, size - 2)
    queue = [start]
    visited = set()
    steps = 0
    distance = np.zeros((size, size), dtype=np.uint8)
    distance[start] = 0
    while len(queue) > 0:
        current = queue.pop(0)
        if current == goal and distance[goal] >= min_steps:
            print(min_steps)
            print_maze(maze)
            print(distance)
            return True
        if current in visited:
            continue
        visited.add(current)
        x, y = current
        if maze[x, y] == 2:
            continue
        if x - 1 >= 0 and maze[x - 1, y] != 1:
            queue.append((x - 1, y))
            if maze[x-1, y] != 2 and distance[x-1, y] == 0:
                distance[x - 1, y] = distance[x, y] + 1
        if x + 1 < size and maze[x + 1, y] != 1:
            queue.append((x + 1, y))
            if maze[x+1, y] != 2 and distance[x+1, y] == 0:
                distance[x + 1, y] = distance[x, y] + 1
        if y - 1 >= 0 and maze[x, y - 1] != 1:
            queue.append((x, y - 1))
            if maze[x, y-1] != 2 and distance[x, y-1] == 0:
                distance[x, y - 1] = distance[x, y] + 1
        if y + 1 < size and maze[x, y + 1] != 1:
            queue.append((x, y + 1))
            if maze[x, y+1] != 2 and distance[x, y+1] == 0:
                distance[x, y + 1] = distance[x, y] + 1
    # print distance matrix
    return False

def print_maze(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            print(x[i, j], end=' ')
        print()

def check_maze(x):
    assert x[0, 0] == 1
    assert x[-1, -1] == 1
    assert x[1, 1] == 0
    assert x[-2, -2] == 3
    assert x[0, 1] == 1
    assert x[1, 0] == 1
    assert x[-1, -2] == 1
    assert x[-2, -1] == 1
    for i in range(1, x.shape[0] - 1):
        for j in range(1, x.shape[1] - 1):
            assert x[i, j] in [0, 1, 2, 3]
    return True
def check_solvability_maze(maze):
    # check if maze is solvable using BFS
    size = maze.shape[0]
    start = (1, 1)
    goal = (size - 2, size - 2)
    queue = [start]
    visited = set()
    while len(queue) > 0:
        current = queue.pop(0)
        if current == goal:
            return True
        if current in visited:
            continue
        visited.add(current)
        x, y = current
        if maze[x, y] == 2:
            continue
        if x - 1 >= 0 and maze[x - 1, y] != 1:
            queue.append((x - 1, y))
        if x + 1 < size and maze[x + 1, y] != 1:
            queue.append((x + 1, y))
        if y - 1 >= 0 and maze[x, y - 1] != 1:
            queue.append((x, y - 1))
        if y + 1 < size and maze[x, y + 1] != 1:
            queue.append((x, y + 1))
    return False

if __name__ == '__main__':
    solvable = False
    num = 0
    while not solvable:
        seed = np.random.randint(0, 10000)
        maze = random_sutton_maze2(9, 3, seed = seed)
        solvable = check_solvability_and_steps_maze2(maze, 10, 20)
        num += 1
        if num % 100 == 0:
            print("Number of attempts: {}".format(num))
        if num > 10000:
            print("Could not find solvable maze")
            break
    print_maze(maze)
    assert check_maze(maze)
    assert check_solvability_maze(maze)

