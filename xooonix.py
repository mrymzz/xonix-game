import time
import random
import numpy as np
from random import randint
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys

sys.setrecursionlimit(10**6)

####################################
########### constants ##############
####################################
GRID_DIVISION = 40
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800

# WINDOW_WIDTH1 = 760
# WINDOW_HEIGHT1 = 560
INTERVAL = 1

RIGHT = (1, 0)
LEFT = (-1, 0)
TOP = (0, 1)
BOTTOM = (0, -1)
newcolor = 1
oldColor = 0

####################################
########### game state #############
####################################

grid = [[0 for j in range(GRID_DIVISION)] for i in range(GRID_DIVISION)]
grid2 = [[0 for j in range(GRID_DIVISION)] for i in range(GRID_DIVISION)]


current_delta = RIGHT

# chech lose and run
lose = False
run = False

# position
x = GRID_DIVISION // 2
y = 1

# DIRECTION

L = False
R = False
U = True
D = False
change_direction = 0
count = 0
#
in_path = False
floodPoint_x = 0
floodPoint_y = 0
ball_in_area = False
ball_in_area2 = False

# changing direction
U_R = False
D_R = False
U_L = False
D_L = False
L_U = False
R_U = False
L_D = False
R_D = False

# area_vertecies = []
# i and j for flood_fill
i = 0
j = 0
# border
border = []
# path
path = []


BALL_RADIUS = 20
BALL_COLOR = (1, 0, 0)
BALL_SPEED = 1
NUM_BALLS = 2
cell_width = WINDOW_WIDTH / GRID_DIVISION
cell_height = WINDOW_HEIGHT / GRID_DIVISION
balls = []
for i in range(NUM_BALLS):
    ball_x = random.randint(2, GRID_DIVISION - 2)
    print(ball_x)
    ball_y = random.randint(2, GRID_DIVISION - 2)
    print(ball_y)
    ball_direction = [random.choice([-1, 1]), random.choice([-1, 1])]
    balls.append(
        [
            ball_x * cell_width,
            ball_y * cell_height,
            BALL_RADIUS,
            BALL_COLOR,
            BALL_SPEED,
            ball_direction,
        ]
    )


def draw_ball(ball):
    glBegin(GL_POLYGON)
    glColor3f(*ball[3])
    for i in range(360):
        rad = i * np.pi / 180
        x = (ball[0]) + (ball[2] * np.cos(rad))
        y = (ball[1]) + (ball[2] * np.sin(rad))
        glVertex2f(x, y)
    glEnd()


def draw_balls():
    for ball in balls:
        draw_ball(ball)


def update_balls():
    global lose
    for i in range(NUM_BALLS):
        ball1 = balls[i]
        ball1[0] += ball1[4] * ball1[5][0]
        ball1[1] += ball1[4] * ball1[5][1]

        # check if ball hits the walls
        if (
            ball1[0] - ball1[2] == cell_width
            or ball1[0] + ball1[2] == WINDOW_WIDTH - cell_width
        ):
            ball1[5][0] *= -1
        if (
            ball1[1] - ball1[2] == cell_width
            or ball1[1] + ball1[2] == WINDOW_HEIGHT - cell_width
        ):
            ball1[5][1] *= -1

        # check if ball intersects with path
        for cell in path:
            cx, cy = (
                cell[0] * cell_width + cell_width / 2,
                cell[1] * cell_height + cell_height / 2,
            )
            dx, dy = ball1[0] - cx, ball1[1] - cy
            distance = np.sqrt(dx**2 + dy**2)
            if distance < ball1[2] + cell_width / 2:
                lose = True
        # collision with border
        for cell in border:
            cx, cy = (
                cell[0] * cell_width + cell_width / 2,
                cell[1] * cell_height + cell_height / 2,
            )
            dx, dy = ball1[0] - cx, ball1[1] - cy
            distance = np.sqrt(dx**2 + dy**2)
            if distance < ball1[2] + cell_width / 2:
                ball1[5][0] *= -1
                ball1[5][1] *= -1

        # check for collision with other balls
        for j in range(i + 1, NUM_BALLS):
            ball2 = balls[j]
            dx, dy = ball1[0] - ball2[0], ball1[1] - ball2[1]
            distance = np.sqrt(dx**2 + dy**2)
            if distance < ball1[2] + ball2[2]:
                # calculate new velocities after collision
                v1 = np.array(ball1[5]) * ball1[4]
                v2 = np.array(ball2[5]) * ball2[4]
                m1 = np.pi * ball1[2] ** 2  # assuming density is 1
                m2 = np.pi * ball2[2] ** 2
                v1_new = v1 - 2 * m2 / (m1 + m2) * np.dot(
                    v1 - v2, np.array(ball1[0:2]) - np.array(ball2[0:2])
                ) / distance**2 * (np.array(ball1[0:2]) - np.array(ball2[0:2]))
                v2_new = v2 - 2 * m1 / (m1 + m2) * np.dot(
                    v2 - v1, np.array(ball2[0:2]) - np.array(ball1[0:2])
                ) / distance**2 * (np.array(ball2[0:2]) - np.array(ball1[0:2]))
                ball1[5] = list(v1_new / np.linalg.norm(v1_new))
                ball2[5] = list(v2_new / np.linalg.norm(v2_new))
                ball1[4] = np.linalg.norm(v1_new)
                ball2[4] = np.linalg.norm(v2_new)

                # move the balls apart
                overlap_vec = get_overlap_vector(ball1, ball2)
                if overlap_vec is not None:
                    ball1[0] -= overlap_vec[0] * 0.5
                    ball1[1] -= overlap_vec[1] * 0.5
                    ball2[0] += overlap_vec[0] * 0.5
                    ball2[1] += overlap_vec[1] * 0.5


def get_overlap_vector(ball1, ball2):
    """
    Returns the minimum translation vector needed to separate two balls.
    If the balls do not overlap, returns None.
    """
    dx, dy = ball1[0] - ball2[0], ball1[1] - ball2[1]
    distance = np.sqrt(dx**2 + dy**2)
    if distance >= ball1[2] + ball2[2]:
        return None

    # get the axes of separation
    axes = []
    for angle in range(0, 360, 10):
        axes.append(np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))]))

    # project the balls onto the axes and check for overlap
    overlap = float("inf")
    overlap_axis = None
    for axis in axes:
        projection1 = project_onto_axis(ball1, axis)
        projection2 = project_onto_axis(ball2, axis)
        if not overlap_on_axis(projection1, projection2):
            return None
        else:
            o = get_overlap_on_axis(projection1, projection2)
            if o < overlap:
                overlap = o
                overlap_axis = axis

    # calculate the minimum translation vector
    mtv = overlap_axis * overlap
    return mtv


def project_onto_axis(ball, axis):
    """
    Projects a ball onto an axis and returns the min and max projection values.
    """
    center = np.array(ball[:2])
    radius = ball[2]
    projection = np.dot(center, axis)
    min_proj = projection - radius
    max_proj = projection + radius
    return (min_proj, max_proj)


def overlap_on_axis(projection1, projection2):
    """
    Checks if two 1D projections overlap.
    """
    return (projection1[0] <= projection2[1]) and (projection2[0] <= projection1[1])


def get_overlap_on_axis(projection1, projection2):
    """
    Returns the overlap distance between two 1D projections.
    Assumes that the projections overlap.
    """
    return min(projection1[1], projection2[1]) - max(projection1[0], projection2[0])


# points of walls
for i in range(GRID_DIVISION):
    border.append((i, 0))
    grid[i][0] = newcolor
    grid2[i][0] = newcolor
    border.append((i, GRID_DIVISION - 1))
    grid2[i][GRID_DIVISION - 1] = newcolor
for j in range(GRID_DIVISION):
    border.append((0, j))
    grid[i][0] = newcolor
    grid2[i][0] = newcolor
    border.append((GRID_DIVISION - 1, j))
    grid[i][GRID_DIVISION - 1] = newcolor
    grid2[i][GRID_DIVISION - 1] = newcolor


####################################
######## graphics helpers ##########
####################################
def init():
    # glClearColor(0.0, 0.0, 0.0, 0.0)

    glMatrixMode(GL_PROJECTION)  # ortho or perspective NO BRAINER
    glLoadIdentity()
    glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, 0, 1)  # l,r,b,t,n,f
    glMatrixMode(GL_MODELVIEW)


def draw_cell(i, j):
    cell_width = WINDOW_WIDTH / GRID_DIVISION
    cell_height = WINDOW_HEIGHT / GRID_DIVISION
    glBegin(GL_QUADS)
    glVertex2d(cell_width * i, cell_height * j)
    glVertex2d(cell_width * (i + 1), cell_height * j)
    glVertex2d(cell_width * (i + 1), cell_height * (j + 1))
    glVertex2d(cell_width * i, cell_height * (j + 1))
    glEnd()


# def dfs(grid, i, j, old_color, new_color):
#     n = 39
#     m = 39
#     if i <= 0 or i >= n or j <= 0 or j > m or grid[i][j] != old_color:
#         return
#     else:
#         grid[i][j] = new_color
#         dfs(grid, i + 1, j, old_color, new_color)
#         dfs(grid, i - 1, j, old_color, new_color)
#         dfs(grid, i, j + 1, old_color, new_color)
#         dfs(grid, i, j - 1, old_color, new_color)


def dfs(grid, i, j, old_color, new_color):
    n = len(grid) - 1
    m = len(grid[0]) - 1
    stack = [(i, j)]
    while stack:
        x, y = stack.pop()
        if x <= 0 or x >= n or y <= 0 or y >= m or grid[x][y] != old_color:
            continue
        grid[x][y] = new_color
        stack.append((x + 1, y))
        stack.append((x - 1, y))
        stack.append((x, y + 1))
        stack.append((x, y - 1))


def flood_fill(grid, i, j, new_color):
    old_color = grid[i][j]
    if old_color == new_color:
        return
    dfs(grid, i, j, old_color, new_color)


def draw_closedArea(grid):
    global newcolor, balls
    for i in range(GRID_DIVISION):
        for j in range(GRID_DIVISION):
            if grid[i][j] == newcolor:
                glColor(1, 0, 0)
                draw_cell(i, j)


# check if the ball ion the area flooded or not
def check_ball(grid):
    global newcolor, balls, cell_width, cell_height, ball_in_area
    for i in range(GRID_DIVISION):
        for j in range(GRID_DIVISION):
            if grid[i][j] == newcolor and (i, j) not in border:
                for ball in balls:
                    cx, cy = (
                        i * cell_width + cell_width / 2,
                        j * cell_height + cell_height / 2,
                    )
                    dx, dy = ball[0] - cx, ball[1] - cy
                    distance = np.sqrt(dx**2 + dy**2)
                    if distance < ball[2] + cell_height / 2:
                        ball_in_area = True
                        print("ball in area ")


def check_area2():
    global i, j, floodPoint_x, floodPoint_y, ball_in_area2, ball_in_area, L, R, U, D, L_U, L_D, R_U, R_D, U_R, U_L, D_R, D_L, in_path
    print("i ,j before ", i, j)
    if in_path:
        in_path = False
        print("pathhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
        if L or R:
            if y >= GRID_DIVISION // 2:
                j += 2
            else:
                j -= 2
        elif U or D:
            if x >= GRID_DIVISION // 2:
                i += 2
            else:
                i -= 2
    else:
        if U_L:
            if grid[i + 1][j] == 1:
                i = floodPoint_x
                j = floodPoint_y + 1

            else:
                i = floodPoint_x + 1
                j = floodPoint_y

        elif D_L:
            if grid[i][j - 1] == 1:
                i = floodPoint_x + 1
                j = floodPoint_y
            else:
                i = floodPoint_x
                j = floodPoint_y - 1

        elif U_R:
            if grid[i - 1][j] == 1:
                i = floodPoint_x
                j = floodPoint_y + 1
            else:
                i = floodPoint_x - 1
                j = floodPoint_y

        elif D_R:
            if grid[i - 1][j] == 1:
                i = floodPoint_x
                j = floodPoint_y - 1
            else:
                i = floodPoint_x - 1
                j = floodPoint_y
        elif R_U:
            if grid[i + 1][j] == 1:
                i = floodPoint_x
                j = floodPoint_y - 1
            else:
                i = floodPoint_x + 1
                j = floodPoint_y

        elif L_U:
            if grid[i - 1][j] == 1:
                i = floodPoint_x
                j = floodPoint_y - 1
            else:
                i = floodPoint_x - 1
                j = floodPoint_y

        elif R_D:
            if grid[i + 1][j] == 1:
                i = floodPoint_x
                j = floodPoint_y + 1
            else:
                i = floodPoint_x + 1
                j = floodPoint_y

        elif L_D:
            if grid[i - 1][j] == 1:
                i = floodPoint_x
                j = floodPoint_y + 1
            else:
                i = floodPoint_x - 1
                j = floodPoint_y

    print("i ,j after module", i, j)
    flood_fill(grid2, i, j, newcolor)
    check_ball(grid2)
    for a in range(GRID_DIVISION):
        for b in range(GRID_DIVISION):
            grid2[a][b] = grid[a][b]

    if ball_in_area:
        ball_in_area2 = True
        ball_in_area = False


def draw_border():
    for i, j in border:
        draw_cell(i, j)


def draw_path():
    for i, j in path:
        draw_cell(i, j)


def draw_fan(a, b):
    glColor3f(1.0, 0, 1.0)
    draw_cell(a, b)


def display():
    global path, border, lose, grid, grid2, newcolor, L, R, U, D, i, j, ball_in_area, ball_in_area2, x, y, count, in_path, R_U, R_D, L_U, L_D, U_R, D_R, U_L, D_L, floodPoint_x, floodPoint_y
    glClear(GL_COLOR_BUFFER_BIT)
    # glClearColor(0, 0, 0, 0)
    count += 1
    draw_balls()
    if not lose:
        update_balls()

    # check losing

    if (x, y) in path:
        if (x, y) not in border:
            lose = True
    # Check if the fan has touched the border
    if (x, y) in border:
        for point in path:
            if point not in border:
                border.append(point)
                grid[point[0]][point[1]] = newcolor
                grid2[point[0]][point[1]] = newcolor

        if len(path) >= 2:
            i = (path[0][0] + path[-1][0]) // 2
            j = (path[0][1] + path[-1][1]) // 2
            print("the point of flood is", i, j)
        for point in path:
            if i == point[0] and j == point[1]:
                in_path = True
                if L or R:
                    if y >= (GRID_DIVISION) // 2:
                        j -= 1

                    else:
                        j += 1

                elif U or D:
                    if x >= (GRID_DIVISION) // 2:
                        i -= 1

                    else:
                        i += 1

        if grid[i][j] == 1:
            in_path = False
            if U_L:
                i = floodPoint_x - 1
                j = floodPoint_y - 1

            elif D_L:
                i = floodPoint_x - 1
                j = floodPoint_y + 1
                print("D_L")
            elif U_R:
                i = floodPoint_x + 1
                j = floodPoint_y - 1
                print("U_R")

            elif D_R:
                i = floodPoint_x + 1
                j = floodPoint_y + 1
                print("U_R")
            elif R_U:
                i = floodPoint_x - 1
                j = floodPoint_y + 1
                print("R_U")

            elif L_U:
                i = floodPoint_x + 1
                j = floodPoint_y + 1
                print("L_U")

            elif R_D:
                i = floodPoint_x - 1
                j = floodPoint_y - 1
                print("R_D")

            elif L_D:
                i = floodPoint_x + 1
                j = floodPoint_y - 1

        for point in path:
            if i == point[0] and j == point[1]:
                in_path = True
                if L or R:
                    if y >= (GRID_DIVISION) // 2:
                        j -= 1

                    else:
                        j += 1

                elif U or D:
                    if x >= (GRID_DIVISION) // 2:
                        i -= 1

                    else:
                        i += 1

        path.clear()
        if count <= 1:
            flood_fill(grid2, i, j, newcolor)
            check_ball(grid2)
            for a in range(GRID_DIVISION):
                for b in range(GRID_DIVISION):
                    grid2[a][b] = grid[a][b]

            if ball_in_area:
                ball_in_area = False
                # ball_in_area2 = False
                check_area2()
                if ball_in_area2:
                    i = 0
                    j = 0
                    ball_in_area2 = False

            print("i,j before flooding", i, j)
            flood_fill(grid, i, j, newcolor)
            for a in range(GRID_DIVISION):
                for b in range(GRID_DIVISION):
                    grid2[a][b] = grid[a][b]
        draw_closedArea(grid)
        # if not ball_in_area:
        # print(i, j)
        glColor(1, 1, 0)  # Set color to yellow
        draw_border()
        glColor(1, 1, 1)
        draw_path()
        glColor(1, 1, 0)
        draw_fan(x, y)

    else:
        count = 0
        # ball_in_area = False
        glColor(1, 1, 1)
        draw_path()
        glColor(1, 0, 0)
        draw_closedArea(grid)
        glColor(1, 1, 0)  # Set color to white
        draw_border()
        glColor(1, 1, 0)
        draw_fan(x, y)

    glutSwapBuffers()


####################################
############### Timers #############
####################################


def game_timer(v):
    display()
    glutTimerFunc(INTERVAL, game_timer, 1)


####################################
############ Callbacks #############
####################################
def keyboard_callback(key, X, Y):
    global x, y, lose, L, R, U, D, floodPoint_x, floodPoint_y, R_U, R_D, L_U, L_D, U_R, D_R, U_L, D_L

    if not lose:
        path.append((x, y))
        if key == GLUT_KEY_LEFT and x > 0:  # and not R and not L:
            if U or D:
                print(x, y)
                if U:
                    U_L = True
                    D_L = False
                    U_R = False
                    D_R = False
                    R_U = False
                    L_U = False
                    R_D = False
                    L_D = False

                    floodPoint_x = x
                    floodPoint_y = y
                else:
                    U_L = False
                    D_L = True
                    U_R = False
                    D_R = False
                    R_U = False
                    L_U = False
                    R_D = False
                    L_D = False
                    floodPoint_x = x
                    floodPoint_y = y
                print("flooded point is ", floodPoint_x, floodPoint_y)
            L = True
            R = False
            U = False
            D = False
            x -= 1

        elif key == GLUT_KEY_RIGHT and x < GRID_DIVISION - 1:  # and not R and not L:
            if U or D:
                print("changed direction ", x, y)
                if U:
                    U_L = False
                    D_L = False
                    U_R = True
                    D_R = False
                    R_U = False
                    L_U = False
                    R_D = False
                    L_D = False
                    floodPoint_x = x
                    floodPoint_y = y
                else:
                    U_L = False
                    D_L = False
                    U_R = False
                    D_R = True
                    R_U = False
                    L_U = False
                    R_D = False
                    L_D = False

                    floodPoint_x = x
                    floodPoint_y = y
                print("flooded point is ", floodPoint_x, floodPoint_y)
            L = False
            R = True
            U = False
            D = False
            x += 1
        elif key == GLUT_KEY_UP and y < GRID_DIVISION - 1:  # and not U and not D:
            if L or R:
                print(x, y)
                if R:
                    U_L = False
                    D_L = False
                    U_R = False
                    D_R = False
                    R_U = True
                    L_U = False
                    R_D = False
                    L_D = False
                    floodPoint_x = x
                    floodPoint_y = y
                else:
                    U_L = False
                    D_L = False
                    U_R = False
                    D_R = False
                    R_U = False
                    L_U = True
                    R_D = False
                    L_D = False
                    floodPoint_x = x
                    floodPoint_y = y
                print("flooded point is ", floodPoint_x, floodPoint_y)
            L = False
            R = False
            U = True
            D = False
            y += 1

        elif key == GLUT_KEY_DOWN and y > 0:  # and not U and not D:
            if L or R:
                print(x, y)
                if R:
                    U_L = False
                    D_L = False
                    U_R = False
                    D_R = False
                    R_U = False
                    L_U = False
                    R_D = True
                    L_D = False
                    floodPoint_x = x
                    floodPoint_y = y
                else:
                    U_L = False
                    D_L = False
                    U_R = False
                    D_R = False
                    R_U = False
                    L_U = False
                    R_D = False
                    L_D = True
                    floodPoint_x = x
                    floodPoint_y = y
                print("flooded point is ", floodPoint_x, floodPoint_y)
            L = False
            R = False
            U = False
            D = True
            y -= 1


if __name__ == "__main__":
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT)
    glutInitWindowPosition(0, 0)
    glutCreateWindow(b"airxonix")
    glutDisplayFunc(display)
    glutTimerFunc(INTERVAL, game_timer, 1)
    glutSpecialFunc(keyboard_callback)
    init()
    glutMainLoop()
