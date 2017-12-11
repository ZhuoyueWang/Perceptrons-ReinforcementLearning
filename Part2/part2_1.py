import numpy as np
import mpmath as math
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import deque
import time
import random
import sys
from PyQt5 import QtGui, QtCore
import csv

def training(Q,initial,Qfail,N,Nfail,grid):
    state = initial
    paddle_height = 0.2
    count = 0
    # parameters C and gamma
    C = 500

    gamma = 0.7

    #initial state
    ball_x = state[0]
    ball_y = state[1]
    velocity_x = state[2]
    velocity_y = state[3]
    paddle_y = state[4]
    Ne = 5
    # find Q(s,a)
    idx1 = determine_cell(ball_x,grid)
    idx2 = determine_cell(ball_y,grid)

    if velocity_x > 0:
        idx3 = 0
    else:
        idx3 = 1
    if abs(velocity_y) < 0.015:
        idx4 = 0
    elif velocity_y > 0:
        idx4 = 1
    else:
        idx4 = 2
    if paddle_y == 1 - paddle_height:
        idx5 = grid-1
    else:
        idx5 = int(np.floor(grid*paddle_y/(1-paddle_height)))

    a = np.argmax(Q[idx1][idx2][idx3][idx4][idx5])

    while (not over(state)):
        Rs = 0
        alpha = C / (C + N[idx1][idx2][idx3][idx4][idx5][a])
        N[idx1][idx2][idx3][idx4][idx5][a] += 1

        # find successor
        if a == 1:
            paddle_y += 0.04
        elif a == 2:
            paddle_y += -0.04
        if paddle_y < 0:
            paddle_y = 0
        if paddle_y > 1 - paddle_height:
            paddle_y = 1 - paddle_height
        ball_x += velocity_x
        ball_y += velocity_y
        if ball_y < 0 :
            ball_y = -ball_y
            velocity_y = -velocity_y
        if ball_y > 1:
            ball_y = 2-ball_y
            velocity_y = -velocity_y
        if ball_x < 0:
            ball_x = - ball_x
            velocity_x = -velocity_x
        if ball_x >= 1 and ball_y >= paddle_y and ball_y <= paddle_y + paddle_height: # if bounce
            ball_x = 2 - ball_x
            U = random.uniform(-0.015,0.015)
            V = random.uniform(-0.03,0.03)
            velocity_x = -velocity_x + U
            velocity_y = -velocity_y + V
            if abs(velocity_x) < 0.03:
                velocity_x = velocity_x - U
            if abs(velocity_x) > 1:
                velocity_x = velocity_x - U
            if abs(velocity_y) > 1:
                velocity_y = velocity_y - V
            count += 1
            Rs = 1
        state = [ball_x,ball_y,velocity_x,velocity_y,paddle_y]

        # TD update
        if not over(state):
            idx1_new = determine_cell(ball_x,grid)
            idx2_new = determine_cell(ball_y,grid)
            if velocity_x > 0:
                idx3_new = 0
            else:
                idx3_new = 1
            if abs(velocity_y) < 0.015:
                idx4_new = 0
            elif velocity_y > 0:
                idx4_new = 1
            else:
                idx4_new = 2

            if paddle_y == 1 - paddle_height:
                idx5_new = grid-1
            else:
                idx5_new = int(np.floor(grid*paddle_y/(1-paddle_height)))

            Qs_prime = Q[idx1_new][idx2_new][idx3_new][idx4_new][idx5_new]
            kkk = 0
            for i in range(3):
                if N[idx1_new][idx2_new][idx3_new][idx4_new][idx5_new][i] < Ne and kkk == 0:
                    a_prime = i
                    kkk = 1
            if kkk == 0:
                a_prime = np.argmax(Qs_prime)
            # a_prime = np.argmax(Qs_prime)

            Q[idx1][idx2][idx3][idx4][idx5][a] += alpha * (Rs + gamma *Qs_prime[a_prime] - Q[idx1][idx2][idx3][idx4][idx5][a])
            idx1 = idx1_new
            idx2 = idx2_new
            idx3 = idx3_new
            idx4 = idx4_new
            idx5 = idx5_new
            Qs = Qs_prime
            a = a_prime
            #N[idx1][idx2][idx3][idx4][idx5][a] += 1
        else:
            Rs = -5
            Q[idx1][idx2][idx3][idx4][idx5][a] += alpha * (Rs + gamma * Qfail - Q[idx1][idx2][idx3][idx4][idx5][a])
            Nfail += 1
    #print(count)
    #return Q

def testing(Q,initial,Qfail,N,Nfail,grid):
    state = initial
    paddle_height = 0.2
    count = 0
    #initial state

    ball_x = state[0]
    ball_y = state[1]
    velocity_x = state[2]
    velocity_y = state[3]
    paddle_y = state[4]

    while (not over(state)):


        # find Q(s,a)
        idx1 = determine_cell(ball_x,grid)
        idx2 = determine_cell(ball_y,grid)

        if velocity_x > 0:
            idx3 = 0
        else:
            idx3 = 1

        if abs(velocity_y) < 0.015:
            idx4 = 0
        elif velocity_y > 0:
            idx4 = 1
        else:
            idx4 = 2
        if paddle_y == 1 - paddle_height:
            idx5 = grid-1
        else:
            idx5 = int(np.floor(grid * paddle_y / (1 - paddle_height)))
        Qs = Q[idx1][idx2][idx3][idx4][idx5]
        a = np.argmax(Qs)

        # find successor
        if a == 1:
            paddle_y += 0.04
        elif a == 2:
            paddle_y += -0.04
        if paddle_y < 0:
            paddle_y = 0
        if paddle_y > 1 - paddle_height:
            paddle_y = 1 - paddle_height
        ball_x += velocity_x
        ball_y += velocity_y

        # special conditions
        if ball_y < 0 :
            ball_y = -ball_y
            velocity_y = -velocity_y
        if ball_y > 1:
            ball_y = 2-ball_y
            velocity_y = -velocity_y
        if ball_x < 0:
            ball_x = - ball_x
            velocity_x = -velocity_x
        if ball_x >= 1 and ball_y >= paddle_y and ball_y <= paddle_y + paddle_height: # if bounce
            ball_x = 2 - ball_x
            U = random.uniform(-0.015,0.015)
            V = random.uniform(-0.03,0.03)
            velocity_x = -velocity_x + U
            velocity_y = -velocity_y + V
            if abs(velocity_x) < 0.03 or abs(velocity_x) > 1:
                velocity_x = velocity_x - U
            # count number of bounce
            count += 1

        state = [ball_x,ball_y,velocity_x,velocity_y,paddle_y]

    #print(count)
    return count

def write_csv(Q,initial,grid):
    state = initial
    paddle_height = 0.2
    count = 0
    # initial state

    ball_x = state[0]
    ball_y = state[1]
    velocity_x = state[2]
    velocity_y = state[3]
    paddle_y = state[4]
    with open('anime_input.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(state)
        while (not over(state)):

            # find Q(s,a)
            idx1 = determine_cell(ball_x,grid)
            idx2 = determine_cell(ball_y,grid)

            if velocity_x > 0:
                idx3 = 0
            else:
                idx3 = 1
            if abs(velocity_y) < 0.015:
                idx4 = 0
            elif velocity_y > 0:
                idx4 = 1
            else:
                idx4 = 2
            if paddle_y == 1 - paddle_height:
                idx5 = grid-1
            else:
                idx5 = int(np.floor(grid * paddle_y / (1 - paddle_height)))
            Qs = Q[idx1][idx2][idx3][idx4][idx5]
            a = np.argmax(Qs)

            # find successor
            if a == 1:
                paddle_y += 0.04
            elif a == 2:
                paddle_y += -0.04
            if paddle_y < 0:
                paddle_y = 0
            if paddle_y > 1 - paddle_height:
                paddle_y = 1 - paddle_height
            ball_x += velocity_x
            ball_y += velocity_y

            # special conditions
            if ball_y < 0:
                ball_y = -ball_y
                velocity_y = -velocity_y
            if ball_y > 1:
                ball_y = 2 - ball_y
                velocity_y = -velocity_y
            if ball_x < 0:
                ball_x = - ball_x
                velocity_x = -velocity_x
            if ball_x >= 1 and ball_y >= paddle_y and ball_y <= paddle_y + paddle_height:  # if bounce
                ball_x = 2 - ball_x
                U = random.uniform(-0.015, 0.015)
                V = random.uniform(-0.03, 0.03)
                velocity_x = -velocity_x + U
                velocity_y = -velocity_y + V
                if abs(velocity_x) < 0.03 or abs(velocity_x) > 1:
                    velocity_x = velocity_x - U
                # count number of bounce
                count += 1

            state = [ball_x, ball_y, velocity_x, velocity_y, paddle_y]
            writer.writerow(state)
            #print(count)


def over(state): # check if the game fails: return True if failure
    ball_x = state[0]
    ball_y = state[1]
    paddle_y = state[4]
    if (ball_x > 1 and  (ball_y > paddle_y + 0.2 or ball_y < paddle_y)): # the ball passes the paddle
        return True
    else:
        return False


def determine_cell(x,grid): # determine where the ball is in the 12*12 grids
    for i in range(grid):
        if x <= (i+1)/grid*1.0:
            return i

def determine_cell(x): # determine where the ball is in the 12*12 grids
    for i in range(12):
        if x <= (i+1)/12.0:
            return i


def main():
    grid = 12

    # idx1 x; idx2 y; idx3 vx  +  -  ; idx4 vy  0  +  -  ; idx5 paddle y
    # three actions [no move, up, down]
    Q = [[[[[[0,0,0] for x in range(grid)] for y in range(3)] for z in range(2)] for k in range(grid)] for l in range(grid)]
    Qfail = 0
    # appearance of the same state
    N = [[[[[[0,0,0] for x in range(grid)] for y in range(3)] for z in range(2)] for k in range(grid)] for l in range(grid)]
    Nfail = 0
    counts = []
    print(np.shape(Q))
    paddle_height = 0.2
    initial = [0.5,0.5,0.03,0.01,0.5-paddle_height/2,0.5-paddle_height/2]
    # training Q
    training_times = 100000

    for times in range(training_times):
        training(Q,initial,Qfail,N,Nfail,grid)

    # test
    test_times = 1000
    for times in range(test_times):
        count = testing(Q,initial,Qfail,N,Nfail,grid)
        counts.append(count)
    #print("the Q matrix is")
    #print(Q)
    #print(N)
    print("the average number of bounce off is: ")
    #print(counts)
    print(np.average(counts))

    # show an animation
    write_csv(Q,initial,grid)
if __name__== "__main__":
  main()
