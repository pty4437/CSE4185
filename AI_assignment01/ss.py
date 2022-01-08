###### Write Your Library Here ###########

import sys
import queue
import math
import itertools
from collections import deque


#########################################


def search(maze, func):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_four_circles": astar_four_circles,
        "astar_many_circles": astar_many_circles
    }.get(func)(maze)


# -------------------- Stage 01: One circle - BFS Algorithm ------------------------ #

def bfs(maze):
    """
    [문제 01] 제시된 stage1의 맵 세가지를 BFS Algorithm을 통해 최단 경로를 return하시오.(20점)
    """

    a, b = maze.getDimensions()
    visit = [[0 for col in range(b)] for row in range(a)]

    start_point = maze.startPoint()

    path = []
    visit[start_point[0]][start_point[1]] = 1

    ####################### Write Your Code Here ################################

    queue = deque()
    queue.append((start_point[0], start_point[1]))

    desX, desY = maze.circlePoints()[0]

    while queue:
        row, col = queue.popleft()

        if row == desX:
            if col == desY:
                print("shortest path is", visit[row][col])
                break

        for x, y in maze.neighborPoints(row, col):
            if visit[x][y] == 0:
                visit[x][y] = visit[row][col] + 1
                queue.append((x, y))

    x, y = desX, desY
    path.append((x, y))

    while True:
        if (x, y) == start_point:
            break

        for tmpX, tmpY in maze.neighborPoints(x, y):
            if visit[x][y] - 1 == visit[tmpX][tmpY]:
                path.append((tmpX, tmpY))
                x, y = tmpX, tmpY

    path.reverse()
    # path.append((5,2))

    return path

    ############################################################################


class Node:
    def __init__(self, parent, location):
        self.parent = parent
        self.location = location  # 현재 노드

        self.obj = []

        # F = G+H
        self.f = 0
        self.g = 0
        self.h = 0

    def __eq__(self, other):
        return self.location == other.location and str(self.obj) == str(other.obj)

    def __le__(self, other):
        return self.g + self.h <= other.g + other.h

    def __lt__(self, other):
        return self.g + self.h < other.g + other.h

    def __gt__(self, other):
        return self.g + self.h > other.g + other.h

    def __ge__(self, other):
        return self.g + self.h >= other.g + other.h


# -------------------- Stage 01: One circle - A* Algorithm ------------------------ #

def manhatten_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def astar(maze):
    """
    [문제 02] 제시된 stage1의 맵 세가지를 A* Algorithm을 통해 최단경로를 return하시오.(20점)
    (Heuristic Function은 위에서 정의한 manhatten_dist function을 사용할 것.)
    """

    start_point = maze.startPoint()

    end_point = maze.circlePoints()[0]

    # print(start_point, end_point)

    print("M : ", manhatten_dist(start_point, end_point))

    path = []

    # 일단 bfs 가져옴
    a, b = maze.getDimensions()
    visit = [[0 for col in range(b)] for row in range(a)]

    start_point = maze.startPoint()

    path = []
    visit[start_point[0]][start_point[1]] = 1

    ####################### Write Your Code Here ################################

    q = queue.PriorityQueue()
    q.put((manhatten_dist(start_point, end_point), start_point[0], start_point[1]))

    desX, desY = maze.circlePoints()[0]

    while q:
        realDis, row, col = q.get()
        # print(manhatten_dist((row,col), end_point))

        if row == desX:
            if col == desY:
                print("shortest path is", visit[row][col])
                break

        for x, y in maze.neighborPoints(row, col):
            if visit[x][y] == 0:
                visit[x][y] = visit[row][col] + 1
                q.put((visit[x][y] + manhatten_dist((x, y), end_point), x, y))
                # print(visit[x][y] + manhatten_dist((x,y), end_point))

        # queue.sort

    x, y = desX, desY
    path.append((x, y))

    while True:
        if (x, y) == start_point:
            break

        for tmpX, tmpY in maze.neighborPoints(x, y):
            if visit[x][y] - 1 == visit[tmpX][tmpY]:
                path.append((tmpX, tmpY))
                x, y = tmpX, tmpY

    path.reverse()

    ####################### Write Your Code Here ################################

    return path

    ############################################################################


# -------------------- Stage 02: Four circles - A* Algorithm  ------------------------ #


def stage2_heuristic(maze, p1, p2):
    """
    default = 100

    for x, y in maze.neighborPoints(p1[0], p1[1]):

        #오른쪽
        if x == p1[0] and y == p1[1] + 1:
            if p1[1] < p2[1]:
                default -= 30

        #아래쪽
        if x == p1[0] + 1 and y == p1[1]:
            if p1[0] < p2[0]:
                default -= 30

        #왼쪽
        if x == p1[0] and y == p1[1] - 1:
            if p1[1] > p2[1]:
                default -= 30

        #위쪽
        if x == p1[0] - 1 and y == p1[1]:
            if p1[0] > p2[0]:
                default -= 30



        if p2[0] >= p1[0] and p2[1] >= p1[1]:
            if x == p1[0] + 1 or y == p1[1] + 1:
                default -= 100

        if p2[0] >= p1[0] and p2[1] <= p1[1]:
            if x == p1[0] + 1 or y == p1[1] - 1:
                default -= 100

        if p2[0] <= p1[0] and p2[1] >= p1[1]:
            if x == p1[0] - 1 or y == p1[1] + 1:
                default -= 100

        if p2[0] <= p1[0] and p2[1] <= p1[1]:
            if x == p1[0] - 1 or y == p1[1] - 1:
                default -= 100

    return default"""

    return math.pow(math.dist(p1, p2), 2)
    # return math.dist(p1,p2)


def astar_four_circles(maze):
    """
    [문제 03] 제시된 stage2의 맵 세가지를 A* Algorithm을 통해 최단 경로를 return하시오.(30점)
    (단 Heurstic Function은 위의 stage2_heuristic function을 직접 정의하여 사용해야 한다.)
    """

    end_points = maze.circlePoints()
    end_points.sort()

    path = []

    ####################### Write Your Code Here ################################

    path_now = []

    end_perm = list(itertools.permutations(end_points, 4))

    for k in range(24):
        end_points = end_perm[k]

        path_now = []

        path_one = []
        path_two = []
        path_three = []
        path_four = []

        start_point = maze.startPoint()

        # path = []

        a, b = maze.getDimensions()
        start_point = maze.startPoint()

        # path = []

        for i in range(0, 4):
            # print("시작지점 : ", start_point)
            desX, desY = end_points[i]
            q = queue.PriorityQueue()
            visit = [[0 for col in range(b)] for row in range(a)]
            visit[start_point[0]][start_point[1]] = 1
            q.put((stage2_heuristic(maze, start_point, (desX, desY)), start_point[0], start_point[1]))
            # q.put((manhatten_dist(start_point, (desX, desY)), start_point[0], start_point[1]))

            while q:
                realDis, row, col = q.get()

                if row == desX:
                    if col == desY:
                        # print("shortest path is", visit[row][col])
                        break

                for x, y in maze.neighborPoints(row, col):
                    if visit[x][y] == 0:
                        visit[x][y] = visit[row][col] + 1
                        q.put((visit[x][y] + stage2_heuristic(maze, (x, y), (desX, desY)), x, y))
                        # q.put((visit[x][y] + manhatten_dist((x, y), (desX, desY)), x, y))

            # queue.sort

            x, y = desX, desY
            path_now.append((x, y))

            while True:
                if (x, y) == start_point:
                    break

                for tmpX, tmpY in maze.neighborPoints(x, y):
                    if visit[x][y] - 1 == visit[tmpX][tmpY]:
                        path_now.append((tmpX, tmpY))
                        x, y = tmpX, tmpY

            path_now.reverse()

            if i != 3:
                path_now.pop()

            if i == 0:
                path_one = path_now
            elif i == 1:
                path_two = path_now
            elif i == 2:
                path_three = path_now
            else:
                path_four = path_now

            path_now = []

            start_point = (desX, desY)

        path_now = path_one + path_two + path_three + path_four

        if k == 0 or len(path_now) < len(path):
            path = path_now

        # print(len(path_now))
    # path = path_four

    ####################### Write Your Code Here ################################

    return path

    ############################################################################


# -------------------- Stage 03: Many circles - A* Algorithm -------------------- #

def mst(objectives, edges):
    cost_sum = 0

    ####################### Write Your Code Here ################################

    # print(cost_sum)
    return cost_sum

    ############################################################################


def stage3_heuristic(maze, x, y, visit, des_X, des_Y):
    return 1

    # return mst(obj, len(obj))


def isAllVisit(isVisit):
    for i in range(len(isVisit)):
        if isVisit[i] == False:
            return False

    return True


def prim(len_mat, isVisit):
    res_mat = [[0 for col in range(len(isVisit))] for row in range(len(isVisit))]
    isVisit[0] = True
    count = 1

    for i in range(len(isVisit)):
        for j in range(len(isVisit)):
            res_mat[i][j] = 0

    # while True:
    while True:
        cost = 999

        if count == len(isVisit):
            break

        for i in range(len(isVisit)):
            if isVisit[i] == True:
                for j in range(len(isVisit)):
                    if isVisit[j] == False:
                        if cost >= len_mat[i][j]:
                            x, y, cost = i, j, len_mat[i][j]

        isVisit[y] = True
        count += 1

        res_mat[x][y] = cost
        res_mat[y][x] = cost
        len_mat[x][y] = 999

    return res_mat


def astar_distance(maze, start_point, end_point):
    # print(start_point, end_point)
    # print("M : ", manhatten_dist(start_point, end_point))

    path = []

    # 일단 bfs 가져옴
    a, b = maze.getDimensions()
    visit = [[0 for col in range(b)] for row in range(a)]

    path = []
    visit[start_point[0]][start_point[1]] = 1

    ####################### Write Your Code Here ################################

    q = queue.PriorityQueue()
    q.put((manhatten_dist(start_point, end_point), start_point[0], start_point[1]))

    desX, desY = end_point

    while q:
        realDis, row, col = q.get()
        # print(manhatten_dist((row,col), end_point))

        if row == desX:
            if col == desY:
                # print("shortest path is", visit[row][col])
                break

        for x, y in maze.neighborPoints(row, col):
            if visit[x][y] == 0:
                visit[x][y] = visit[row][col] + 1
                q.put((visit[x][y] + manhatten_dist((x, y), end_point), x, y))
                # print(visit[x][y] + manhatten_dist((x,y), end_point))

        # queue.sort

    x, y = desX, desY
    path.append((x, y))

    while True:
        if (x, y) == start_point or visit[x][y] == 0:
            break

        for tmpX, tmpY in maze.neighborPoints(x, y):

            if visit[x][y] - 1 == visit[tmpX][tmpY]:
                # print(x,y)

                path.append((tmpX, tmpY))
                x, y = tmpX, tmpY

    # print("shortest length : ", len(path))

    return len(path)


def nearPoint(roadMap, len, i, max):
    res = 0

    for k in range(len):
        if roadMap[i][k] != max and roadMap[i][k] != 999 and roadMap[i][k] != 0:
            res += roadMap[i][k]

    # return res
    return res


def print_queue(q):
    tmp_q = []

    while True:

        if q.empty() == True:
            break
        a, b, c = q.get()
        tmp_q.append((a, b, c))

    for d, e, f in tmp_q:
        q.put((d, e, f))


def sum_all_range(roadMap, idx, len):
    res = 0

    for i in range(len):
        res += roadMap[idx][i]

    return res / len


def astar_many_circles(maze):
    """
    [문제 04] 제시된 stage3의 맵 세가지를 A* Algorithm을 통해 최단 경로를 return하시오.(30점)
    (단 Heurstic Function은 위의 stage3_heuristic function을 직접 정의하여 사용해야 하고, minimum spanning tree
    알고리즘을 활용한 heuristic function이어야 한다.)
    """
    end_points = maze.circlePoints()
    end_points.sort()
    path_real_res = []

    for s in range(len(end_points)):

        end_points = maze.circlePoints()
        end_points.sort()

        path = []
        path_result = []

    ####################### Write Your Code Here ################################

        path_tmp = []

        start_point = maze.startPoint()

        a, b = maze.getDimensions()
        start_point = maze.startPoint()

        points = end_points
        points.append(start_point)

        len_mat = [[0 for col in range(len(points))] for row in range(len(points))]
        tmp_mat = [[0 for col in range(len(points))] for row in range(len(points))]
        isVisit = []

        for i in range(len(points)):
            isVisit.append(False)

        for i in range(len(points)):
        # print("aaaa")
            for j in range(len(points)):
                if i == j:
                    tmp_mat[i][j] = 0
                else:
                    if tmp_mat[i][j] != 0:
                        continue
                    tmp_mat[i][j] = astar_distance(maze, points[i], points[j])
                # print("bbbb")
                    tmp_mat[j][i] = tmp_mat[i][j]

        for i in range(len(points)):
            for j in range(len(points)):
                len_mat[i][j] = tmp_mat[i][j]

        road_map = prim(tmp_mat, isVisit)

        maximum = 0
        max = 0

        for i in range(len(points)):
            for j in range(len(points)):
                if i != j:
                    if max < road_map[i][j]:
                        max = road_map[i][j]

        print(max)

        for i in range(len(points)):
            for j in range(len(points)):
                if i != j and road_map[i][j] == 0:
                    road_map[i][j] = max+max/2
                else:
                    if road_map[i][j] > maximum and road_map[i][j] != max+max/2:
                        maximum = road_map[i][j]

        #print("sssssssss", maximum)

        #print(len_mat)
        print(road_map)

        end_queue = queue.PriorityQueue()

        end_queue.put((0, start_point[0], start_point[1]))

        hasRoad = True
        end_point = 0, 0
        index = -1
        min_index = -1
        boolVisit = []
        x = 0
        y = 0

        for i in range(len(isVisit)):
            boolVisit.append(False)

        boolVisit[len(boolVisit) - 1] = True

        count = 0

    # while end_queue:
        for m in range(100):

            if count == len(isVisit) - 1:
                break;

            # print_queue(end_queue)

            min_len = 999999

            hasRoad = False

            for i in range(len(isVisit)):
                if end_points[i] == start_point:
                    index = i
                    break

            if m != 0:
                for i in range(len(isVisit)):
                    if road_map[index][i] != 999 and road_map[index][i] != 0:
                        hasRoad = True
                        """
                        end_queue.put((len_mat[index][i]*2 + 2.3*nearPoint(road_map,len(isVisit),i,max + max/2) + 2*road_map[index][i] ,end_points[i][0], end_points[i][1]))
                        # print(i, nearPoint(road_map, len(isVisit), i))
                        if len_mat[index][i]*2 + 2.3*nearPoint(road_map, len(isVisit), i,max + max/2) + 2*road_map[index][i] < min_len:
                            min_index = i
                            min_len = len_mat[index][i]*2 + 2.3*nearPoint(road_map,len(isVisit),i,max + max/2) + 2*road_map[index][i]"""
                        end_queue.put((len_mat[index][i] + 1.8*nearPoint(road_map, len(isVisit), i,max + max / 2) + road_map[index][i],end_points[i][0], end_points[i][1]))
                        # print(i, nearPoint(road_map, len(isVisit), i))
                        if len_mat[index][i] + 1.8*nearPoint(road_map, len(isVisit), i, max + max / 2) + road_map[index][i] < min_len:
                            min_index = i
                            min_len = len_mat[index][i] + 1.8*nearPoint(road_map, len(isVisit), i, max + max / 2) + road_map[index][i]


            elif m == 0:
                end_point = end_points[s]

            if m != 0:
                if hasRoad == True:
                #start_point = (x,y)
                    end_point = end_points[min_index]

                    for h in range(len(isVisit)):
                        if h != min_index:
                            road_map[h][min_index] = 999
                    road_map[min_index][index] = 999
                else:
                    #tmpDis, tmpX, tmpY = end_queue.get()
                    end_point = (x, y)
                    #start_point = end_point

                if boolVisit[points.index(end_point)] == True:
                    dis, x, y = end_queue.get()
                    continue




            count += 1

            print(start_point, end_point)
            res_point = end_point
            boolVisit[points.index(end_point)] = True

            # print("M : ", manhatten_dist(start_point, end_point))

            path = []

            # 일단 bfs 가져옴
            a, b = maze.getDimensions()
            visit = [[0 for col in range(b)] for row in range(a)]

            path = []
            visit[start_point[0]][start_point[1]] = 1
            real_isVisit = [[0 for col in range(b)] for row in range(a)]

            ####################### Write Your Code Here ################################

            q = queue.PriorityQueue()
            q.put((manhatten_dist(start_point, end_point), start_point[0], start_point[1]))

            desX, desY = end_point

            while q:
                realDis, row, col = q.get()
                # print(manhatten_dist((row,col), end_point))

                if row == desX:
                    if col == desY:
                        # print("shortest path is", visit[row][col])
                        break

                for x, y in maze.neighborPoints(row, col):
                    if visit[x][y] == 0:
                        visit[x][y] = visit[row][col] + 1
                        q.put((visit[x][y] + manhatten_dist((x, y), end_point), x, y))
                        # print(visit[x][y] + manhatten_dist((x,y), end_point))

                # queue.sort

            x, y = desX, desY
            path.append((x, y))

            while True:
                if (x, y) == start_point:
                    break

                for tmpX, tmpY in maze.neighborPoints(x, y):
                    if visit[x][y] - 1 == visit[tmpX][tmpY] and visit[x][y] != 1:
                        path.append((tmpX, tmpY))
                        if maze.isObjective(tmpX,tmpY) == True:
                            if boolVisit[points.index((tmpX,tmpY))] == False:
                                boolVisit[points.index((tmpX,tmpY))] = True
                                count += 1
                            """
                            for h in range(len(isVisit)):
                                road_map[h][points.index((tmpX,tmpY))] = 999
                            road_map[points.index((tmpX,tmpY))][points.index(start_point)] = 999"""
                        x, y = tmpX, tmpY

            start_point = end_point

            path.reverse()
            if count != len(isVisit)-1:
                path.pop()
            path_result += path

        if s == 0:
            path_real_res = path_result
            #path_real_res.append(res_point)
        else:
            if len(path_real_res) > len(path_result):
                print("이게 답이다", end_point)
                path_real_res = path_result
                #path_real_res.append(res_point)

        ####################### Write Your Code Here ################################

    #print(start_point)
    return path_real_res

    ############################################################################
