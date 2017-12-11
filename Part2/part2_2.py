import numpy as np
import mpmath as math
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import deque
import time
import random
import csv



def training(Q,initial,Qfail,N,Nfail):
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
    paddle2_y = state[5]
    Ne = 5
    # find Q(s,a)
    idx1 = determine_cell(ball_x)
    idx2 = determine_cell(ball_y)

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
        idx5 = 11
    else:
        idx5 = int(np.floor(12*paddle_y/(1-paddle_height)))

    if paddle2_y == 1 - paddle_height:
        idx6 = 11
    else:
        idx6 = int(np.floor(12*paddle2_y/(1-paddle_height)))

    a = np.argmax(Q[idx1][idx2][idx3][idx4][idx5][idx6])

    while (over(state) == 0):
        Rs = 0
        alpha = C / (C + N[idx1][idx2][idx3][idx4][idx5][idx6][a])
        N[idx1][idx2][idx3][idx4][idx5][idx6][a] += 1

        # find successor
        if a == 1:
            paddle_y += 0.04
        elif a == 2:
            paddle_y += -0.04

        if paddle2_y+paddle_height/2 - ball_y > 0:
            paddle2_y += -0.02
        elif paddle2_y+paddle_height/2 - ball_y < 0:
            paddle2_y += 0.02


        if paddle_y < 0:
            paddle_y = 0
        if paddle_y > 1 - paddle_height:
            paddle_y = 1 - paddle_height
        if paddle2_y < 0:
            paddle2_y = 0
        if paddle2_y > 1 - paddle_height:
            paddle2_y = 1 - paddle_height

        ball_x += velocity_x
        ball_y += velocity_y
        if ball_y < 0 :
            ball_y = -ball_y
            velocity_y = -velocity_y
        if ball_y > 1:
            ball_y = 2-ball_y
            velocity_y = -velocity_y

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

        if ball_x <= 0 and ball_y >= paddle2_y and ball_y <= paddle2_y + paddle_height: # if bounce
            ball_x = - ball_x
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

        state = [ball_x,ball_y,velocity_x,velocity_y,paddle_y,paddle2_y]

        # TD update
        if not over(state):
            idx1_new = determine_cell(ball_x)
            idx2_new = determine_cell(ball_y)
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
                idx5_new = 11
            else:
                idx5_new = int(np.floor(12*paddle_y/(1-paddle_height)))

            if paddle2_y == 1 - paddle_height:
                idx6_new = 11
            else:
                idx6_new = int(np.floor(12*paddle2_y/(1-paddle_height)))

            Qs_prime = Q[idx1_new][idx2_new][idx3_new][idx4_new][idx5_new][idx6_new]
            kkk = 0
            for i in range(3):
                if N[idx1_new][idx2_new][idx3_new][idx4_new][idx5_new][idx6_new][i] < Ne and kkk == 0:
                    a_prime = i
                    kkk = 1
            if kkk == 0:
                a_prime = np.argmax(Qs_prime)
            # a_prime = np.argmax(Qs_prime)

            Q[idx1][idx2][idx3][idx4][idx5][idx6][a] += alpha * (Rs + gamma *Qs_prime[a_prime] - Q[idx1][idx2][idx3][idx4][idx5][idx6][a])



            idx1 = idx1_new
            idx2 = idx2_new
            idx3 = idx3_new
            idx4 = idx4_new
            idx5 = idx5_new
            idx6 = idx6_new
            Qs = Qs_prime
            a = a_prime
            #N[idx1][idx2][idx3][idx4][idx5][a] += 1
        else:
            Rs = -1
            Q[idx1][idx2][idx3][idx4][idx5][idx6][a] += alpha * (Rs + gamma * Qfail - Q[idx1][idx2][idx3][idx4][idx5][idx6][a])
            Nfail += 1
    #print(count)
    #return Q

def testing(Q,initial,Qfail,N,Nfail):
    state = initial
    paddle_height = 0.2
    count = 0
    #initial state

    ball_x = state[0]
    ball_y = state[1]
    velocity_x = state[2]
    velocity_y = state[3]
    paddle_y = state[4]
    paddle2_y = state[5]


    while (over(state) == 0):


        # find Q(s,a)
        idx1 = determine_cell(ball_x)
        idx2 = determine_cell(ball_y)

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
            idx5 = 11
        else:
            idx5 = int(np.floor(12 * paddle_y / (1 - paddle_height)))

        if paddle2_y == 1 - paddle_height:
            idx6 = 11
        else:
            idx6 = int(np.floor(12*paddle2_y/(1-paddle_height)))

        Qs = Q[idx1][idx2][idx3][idx4][idx5][idx6]
        a = np.argmax(Qs)

        # find successor
        if a == 1:
            paddle_y += 0.04
        elif a == 2:
            paddle_y += -0.04

        if paddle2_y+paddle_height/2 - ball_y > 0:
            paddle2_y += -0.02
        elif paddle2_y+paddle_height/2 - ball_y < 0:
            paddle2_y += 0.02

        if paddle_y < 0:
            paddle_y = 0
        if paddle_y > 1 - paddle_height:
            paddle_y = 1 - paddle_height

        if paddle2_y < 0:
            paddle2_y = 0
        if paddle2_y > 1 - paddle_height:
            paddle2_y = 1 - paddle_height

        ball_x += velocity_x
        ball_y += velocity_y

        # special conditions
        if ball_y < 0 :
            ball_y = -ball_y
            velocity_y = -velocity_y
        if ball_y > 1:
            ball_y = 2-ball_y
            velocity_y = -velocity_y

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

        if ball_x <= 0 and ball_y >= paddle2_y and ball_y <= paddle2_y + paddle_height: # if bounce
            ball_x = - ball_x
            U = random.uniform(-0.015,0.015)
            V = random.uniform(-0.03,0.03)
            velocity_x = -velocity_x + U
            velocity_y = -velocity_y + V
            if abs(velocity_x) < 0.03 or abs(velocity_x) > 1:
                velocity_x = velocity_x - U

        state = [ball_x,ball_y,velocity_x,velocity_y,paddle_y,paddle2_y]

    #print(count)
    if over(state) == 1:
        isWin = 1
    else:
        isWin = 0
    #print(count)
    return count, isWin

def write_csv(Q,initial):
    state = initial
    paddle_height = 0.2
    count = 0
    # initial state

    ball_x = state[0]
    ball_y = state[1]
    velocity_x = state[2]
    velocity_y = state[3]
    paddle_y = state[4]
    paddle2_y = state[5]
    with open('anime_input2.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(state)


        while (not over(state)):

            # find Q(s,a)
            idx1 = determine_cell(ball_x)
            idx2 = determine_cell(ball_y)

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
                idx5 = 11
            else:
                idx5 = int(np.floor(12 * paddle_y / (1 - paddle_height)))

            if paddle2_y == 1 - paddle_height:
                idx6 = 11
            else:
                idx6 = int(np.floor(12 * paddle2_y / (1 - paddle_height)))

            Qs = Q[idx1][idx2][idx3][idx4][idx5][idx6]
            a = np.argmax(Qs)

            # find successor
            if a == 1:
                paddle_y += 0.04
            elif a == 2:
                paddle_y += -0.04

            if paddle2_y + paddle_height / 2 - ball_y > 0:
                paddle2_y += -0.02
            elif paddle2_y + paddle_height / 2 - ball_y < 0:
                paddle2_y += 0.02

            if paddle_y < 0:
                paddle_y = 0
            if paddle_y > 1 - paddle_height:
                paddle_y = 1 - paddle_height

            if paddle2_y < 0:
                paddle2_y = 0
            if paddle2_y > 1 - paddle_height:
                paddle2_y = 1 - paddle_height

            ball_x += velocity_x
            ball_y += velocity_y

            # special conditions
            if ball_y < 0:
                ball_y = -ball_y
                velocity_y = -velocity_y
            if ball_y > 1:
                ball_y = 2 - ball_y
                velocity_y = -velocity_y

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

            if ball_x <= 0 and ball_y >= paddle2_y and ball_y <= paddle2_y + paddle_height:  # if bounce
                ball_x = - ball_x
                U = random.uniform(-0.015, 0.015)
                V = random.uniform(-0.03, 0.03)
                velocity_x = -velocity_x + U
                velocity_y = -velocity_y + V
                if abs(velocity_x) < 0.03 or abs(velocity_x) > 1:
                    velocity_x = velocity_x - U

            state = [ball_x, ball_y, velocity_x, velocity_y, paddle_y, paddle2_y]

            writer.writerow(state)

def over(state): # check if the game fails: return True if failure
    ball_x = state[0]
    ball_y = state[1]
    paddle_y = state[4]
    paddle2_y = state[5]
    if (ball_x > 1 and  (ball_y > paddle_y + 0.2 or ball_y < paddle_y)):
        return -1
    elif (ball_x < 0 and  (ball_y > paddle2_y + 0.2 or ball_y < paddle2_y)): # the ball passes the paddle
        return 1
    else:
        return 0


def determine_cell(x): # determine where the ball is in the 12*12 grids
    for i in range(12):
        if x <= (i+1)/12.0:
            return i


def main():
    grid = 12

    # idx1 x; idx2 y; idx3 vx  +  -  ; idx4 vy  0  +  -  ; idx5 paddle1 y, idx6 paddle2 y
    # three actions [no move, up, down, then paddle2 nomove, up, down]
    Q = [[[[[[[0,0,0] for a in range(12)] for x in range(12)] for y in range(3)] for z in range(2)] for k in range(grid)] for l in range(grid)]
    Qfail = 0
    # appearance of the same state
    N = [[[[[[[0,0,0] for a in range(12)] for x in range(12)] for y in range(3)] for z in range(2)] for k in range(grid)] for l in range(grid)]
    Nfail = 0
    counts = []
    print(np.shape(Q))
    paddle_height = 0.2
    initial = [0.5,0.5,0.03,0.01,0.5-paddle_height/2,0.5-paddle_height/2]
    # training Q
    training_times = 100000
    for times in range(training_times):
        training(Q,initial,Qfail,N,Nfail)
    # test
    test_times = 1000
    winRate = 0
    for times in range(test_times):
        count,isWin = testing(Q,initial,Qfail,N,Nfail)
        counts.append(count)
        winRate += isWin
    #print("the Q matrix is")
    #print(Q)
    #print(N)
    print("the average number of bounce off is: ")
    #print(counts)
    print(np.average(counts))
    print("the winning rate is ")
    print(winRate/test_times)
    write_csv(Q,initial)


if __name__== "__main__":
  main()
