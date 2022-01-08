###### Write Your Library Here ###########

import sys
import queue
import math
import itertools
from collections import deque
import copy
import numpy as np


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
    end_point = maze.circlePoints()[0]

    a, b = maze.getDimensions()
    visit = [[-1 for col in range(b)] for row in range(a)]

    start_point = maze.startPoint()

    path = []
    visit[start_point[0]][start_point[1]] = 1

    ####################### Write Your Code Here ################################

    queue = deque()
    #queue.append((start_point[0], start_point[1]))
    first_nd = Node((-1,-1), start_point)
    first_nd.obj.append(first_nd.location)
    queue.append(first_nd)

    desX, desY = maze.circlePoints()[0]

    while queue:
        pNode = queue.popleft()

        if pNode.location == end_point:
            break

        for x, y in maze.neighborPoints(pNode.location[0], pNode.location[1]):
            if visit[x][y] == -1:
                visit[x][y] = visit[pNode.location[0]][pNode.location[1]] + 1
                node = Node(pNode.location, (x,y))
                node.obj = copy.deepcopy(pNode.obj)
                node.obj.append(node.location)
                queue.append(node)

    path = copy.deepcopy(pNode.obj)

    return path

    ############################################################################


class Node:
    def __init__(self, parent, location):
        self.parent = parent
        self.location = location  # 현재 노드

        self.obj = []
        self.road_mat = []
        self.points = []

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

    path = []

    a, b = maze.getDimensions()
    visit = [[-1 for col in range(b)] for row in range(a)]

    #start_point = maze.startPoint()

    path = []
    visit[start_point[0]][start_point[1]] = 1

    ####################### Write Your Code Here ################################

    q = queue.PriorityQueue()

    first_nd = Node((-1, -1), start_point)
    first_nd.obj.append(first_nd.location)

    q.put((0, first_nd))

    #end_point = maze.circlePoints()[0]

    while q:
        dis, pNode = q.get()
        #print(dis, pNode)

        if pNode.location == end_point:
            break

        for x, y in maze.neighborPoints(pNode.location[0], pNode.location[1]):
            if visit[x][y] == -1:
                visit[x][y] = visit[pNode.location[0]][pNode.location[1]] + 1
                node = Node(pNode.location, (x,y))
                node.obj = copy.deepcopy(pNode.obj)
                node.obj.append(node.location)
                q.put((visit[x][y] + manhatten_dist((x,y), end_point), node))

    path = copy.deepcopy(pNode.obj)

    ####################### Write Your Code Here ################################

    return path

    ############################################################################


# -------------------- Stage 02: Four circles - A* Algorithm  ------------------------ #


def stage2_heuristic(mat, index):
    res_mat = copy.deepcopy(mat)
    res = 0
    count = 0

    for i in range(5):
        if res_mat[0][i] == 0:
            count += 1

    for i in range(5):
        for j in range(5):
            res += res_mat[i][j]

    res = res / 2

    if count == 4:
        res = res * 0.5
    else:
        res = res * (count-1) / count

    for i in range(5):
        for j in range(5):
            if i == index or j == index:
                res_mat[i][j] = 0

    return res_mat, res


def euclidean_distance(p1, p2):
    return math.dist(p1, p2)


def astar_four_circles(maze):
    """
    [문제 03] 제시된 stage2의 맵 세가지를 A* Algorithm을 통해 최단 경로를 return하시오.(30점)
    (단 Heurstic Function은 위의 stage2_heuristic function을 직접 정의하여 사용해야 한다.)
    """

    end_points = maze.circlePoints()
    end_points.sort()

    path = []

    ####################### Write Your Code Here ################################

    dic_obj = {}

    points = copy.deepcopy(end_points)
    points.insert(0, maze.startPoint())

    for i in range(5):
        temp = []
        for j in range(5):
            if i != j:
                # dic_obj[points[i]] = astar_distance(maze, points[i], points[j])
                temp.append(astar_distance(maze, points[i], points[j]))
            else:
                temp.append(0)
        dic_obj[points[i]] = temp

    #print(dic_obj)

    heu_mat = [[0 for col in range(5)] for row in range(5)]

    for i in range(5):
        for j in range(5):
            if i != j:
                heu_mat[i][j] = euclidean_distance(points[i], points[j])

    obj_queue = queue.PriorityQueue()

    first_nd = Node((-1, -1), points[0])

    first_nd.obj.append(points[0])
    first_nd.g = 1
    first_nd.heu_mat = copy.deepcopy(heu_mat)

    obj_queue.put((0, first_nd))

    while True:
        if obj_queue.empty() == True:
            break
        dis, node = obj_queue.get()

        if len(node.obj) == 5:
            break

        for i in range(5):
            if i != points.index(node.location) and points[i] not in node.obj:
                tmp_node = Node(node.location, points[i])
                tmp_node.obj = copy.deepcopy(node.obj)
                tmp_node.obj.append(points[i])
                tmp_node.g = node.g
                tmp_node.g += dic_obj[node.location][i]

                tmp_mat, res = stage2_heuristic(node.heu_mat, i)

                # print(tmp_mat)

                tmp_node.heu_mat = copy.deepcopy(tmp_mat)

                obj_queue.put((res + tmp_node.g, tmp_node))

    path_now = []

    start_point = node.obj[0]

    a, b = maze.getDimensions()

    for i in range(1, 5):
        desX, desY = node.obj[i]
        #q = queue.PriorityQueue()
        visit = [[-1 for col in range(b)] for row in range(a)]
        visit[start_point[0]][start_point[1]] = 1


        q = queue.PriorityQueue()

        first_nd = Node((-1, -1), start_point)
        first_nd.obj.append(first_nd.location)

        q.put((0, first_nd))

        end_point = maze.circlePoints()[0]

        while q:
            dis, pNode = q.get()
            # print(dis, pNode)

            if pNode.location == node.obj[i]:
                break

            for x, y in maze.neighborPoints(pNode.location[0], pNode.location[1]):
                if visit[x][y] == -1:
                    visit[x][y] = visit[pNode.location[0]][pNode.location[1]] + 1
                    tmpnode = Node(pNode.location, (x, y))
                    tmpnode.obj = copy.deepcopy(pNode.obj)
                    tmpnode.obj.append(tmpnode.location)
                    q.put((visit[x][y] + manhatten_dist((x, y), end_point), tmpnode))

        path += pNode.obj

        if i != 4:
            path.pop()

        start_point = node.obj[i]

    ####################### Write Your Code Here ################################

    return path

    ############################################################################


# -------------------- Stage 03: Many circles - A* Algorithm -------------------- #

#def mst(objectives, edges):
    #cost_sum = 0

    ####################### Write Your Code Here ################################

    # print(cost_sum)
    #return cost_sum

    ############################################################################


def isAllVisit(isVisit):
    for i in range(len(isVisit)):
        if isVisit[i] == False:
            return False

    return True


def mst(len_mat, isVisit):
    res_mat = [[0 for col in range(len(isVisit))] for row in range(len(isVisit))]
    isVisit[0] = True
    count = 1

    for i in range(len(isVisit)):
        for j in range(len(isVisit)):
            res_mat[i][j] = 0
    while True:
        cost = 999

        if count == len(isVisit):
            break

        for i in range(len(isVisit)):
            if isVisit[i] == True:
                for j in range(len(isVisit)):
                    if isVisit[j] == False:
                        if cost >= len_mat[i][j] or i == 0:
                            x, y, cost = i, j, len_mat[i][j]

        isVisit[y] = True
        count += 1

        res_mat[x][y] = cost
        res_mat[y][x] = cost
        len_mat[x][y] = 99999

    return res_mat


def astar_distance(maze, start_point, end_point):
    path = []

    a, b = maze.getDimensions()
    visit = [[-1 for col in range(b)] for row in range(a)]

    path = []
    visit[start_point[0]][start_point[1]] = 1

    q = queue.PriorityQueue()

    first_nd = Node((-1, -1), start_point)
    first_nd.obj.append(first_nd.location)

    q.put((0, first_nd))


    while q:
        dis, pNode = q.get()
        # print(dis, pNode)

        if pNode.location == end_point:
            break

        for x, y in maze.neighborPoints(pNode.location[0], pNode.location[1]):
            if visit[x][y] == -1:
                visit[x][y] = visit[pNode.location[0]][pNode.location[1]] + 1
                node = Node(pNode.location, (x, y))
                node.obj = copy.deepcopy(pNode.obj)
                node.obj.append(node.location)
                q.put((visit[x][y] + manhatten_dist((x, y), end_point), node))

    path = copy.deepcopy(pNode.obj)


    return len(path)





def stage3_heuristic(mat):
    tmp_mat = copy.deepcopy(mat)
    res = 0
    back_up = copy.deepcopy(tmp_mat)

    # tmp_mat = np.delete(tmp_mat, index, axis = 0)
    # tmp_mat = np.delete(tmp_mat, index, axis = 1)

    res_mat = copy.deepcopy(tmp_mat)

    isVisit = []

    for i in range(len(tmp_mat)):
        isVisit.append(False)

    tmp_mat = mst(back_up, isVisit)

    for i in range(len(tmp_mat)):
        for j in range(len(tmp_mat)):
            if tmp_mat[i][j] != max:
                res += tmp_mat[i][j]

    res = res / 2 - 1
    # res += 1
    # print(res)

    return res


def astar_many_circles(maze):
    """
    [문제 04] 제시된 stage3의 맵 세가지를 A* Algorithm을 통해 최단 경로를 return하시오.(30점)
    (단 Heurstic Function은 위의 stage3_heuristic function을 직접 정의하여 사용해야 하고, minimum spanning tree
    알고리즘을 활용한 heuristic function이어야 한다.)
    """
    end_points = maze.circlePoints()
    end_points.sort()

    path = []

    ####################### Write Your Code Here ################################

    dic_obj = {}

    points = copy.deepcopy(end_points)
    points.insert(0, maze.startPoint())

    # print(astar_distance(maze, (3,4), (5,3)))

    for i in range(len(points)):
        temp = []
        for j in range(len(points)):
            if i != j:
                # dic_obj[points[i]] = astar_distance(maze, points[i], points[j])
                temp.append(astar_distance(maze, points[i], points[j]) - 1)
            else:
                temp.append(0)
        dic_obj[points[i]] = temp

    tmp_mat = [[0 for col in range(len(points))] for row in range(len(points))]
    isVisit = []

    for i in range(len(points)):
        isVisit.append(False)

    for i in range(len(points)):
        for j in range(len(points)):
            if i == j:
                tmp_mat[i][j] = 0
            else:
                if tmp_mat[i][j] != 0:
                    continue
                # tmp_mat[i][j] = manhatten_dist(points[i], points[j])
                tmp_mat[i][j] = astar_distance(maze, points[i], points[j]) - 1
                # print(tmp_mat[i][j])
                # tmp_mat[i][j] = euclidean_distance(points[i], points[j])
                tmp_mat[j][i] = tmp_mat[i][j]

    back_up = copy.deepcopy(tmp_mat)
    mst_map = mst(tmp_mat, isVisit)

    a, b = maze.getDimensions()

    # max = 0

    # print(mst_map)

    mst_tmp = copy.deepcopy(mst_map)

    obj_queue = queue.PriorityQueue()

    first_nd = Node((-1, -1), points[0])

    first_nd.obj.append(points[0])
    first_nd.g = 1
    first_nd.road_mat = copy.deepcopy(back_up)
    first_nd.points = copy.deepcopy(points)

    # first_nd.road_mat = np.delete(first_nd.road_mat, 0, axis=0)
    # first_nd.road_mat = np.delete(first_nd.road_mat, 0, axis=1)
    # first_nd.obj.append(first_nd.location)

    obj_queue.put((0, first_nd))

    # print(mst_map)

    while True:
        if obj_queue.empty() == True:
            break
        dis, node = obj_queue.get()
        num = 0.0001
        # print(dis, node.g, node.location, node.parent, len(node.obj))

        # print(node.location, node.parent, len(node.obj))

        if len(node.obj) == len(points):
            break

        tmp_mat = copy.deepcopy(node.road_mat)

        res = stage3_heuristic(tmp_mat)

        for i in range(len(node.road_mat)):
            if i != node.points.index(node.location) and node.points[i] not in node.obj:
                tmp_node = Node(node.location, node.points[i])
                tmp_node.obj = copy.deepcopy(node.obj)
                tmp_node.obj.append(node.points[i])
                tmp_node.g = node.g
                tmp_node.g += dic_obj[node.location][points.index(node.points[i])] + 1

                tmp_node.points = copy.deepcopy(node.points)

                tmp_node.points.remove(node.location)

                clip_mat = copy.deepcopy(node.road_mat)
                clip_mat = np.delete(clip_mat, node.points.index(node.location), axis=0)
                clip_mat = np.delete(clip_mat, node.points.index(node.location), axis=1)

                tmp_node.road_mat = copy.deepcopy(clip_mat)

                ####################################
                obj_queue.put((res + tmp_node.g + num, tmp_node))
                num += 0.0001

                ####################################

    path_now = []

    start_point = node.obj[0]

    a, b = maze.getDimensions()

    for i in range(1, len(points)):
        desX, desY = node.obj[i]
        q = queue.PriorityQueue()
        visit = [[-1 for col in range(b)] for row in range(a)]
        visit[start_point[0]][start_point[1]] = 1
        q.put((manhatten_dist(start_point, (desX, desY)), start_point[0], start_point[1]))

        while q:
            # print("???")
            realDis, row, col = q.get()

            if row == desX:
                if col == desY:
                    break

            for x, y in maze.neighborPoints(row, col):
                if visit[x][y] == -1 and (x, y) != start_point:
                    visit[x][y] = visit[row][col] + 1
                    q.put((visit[x][y] + manhatten_dist((x, y), (desX, desY)), x, y))

        x, y = desX, desY
        path_now.append((x, y))

        while True:
            if (x, y) == start_point:
                break
            # print(x,y)

            for tmpX, tmpY in maze.neighborPoints(x, y):
                if visit[x][y] - 1 == visit[tmpX][tmpY]:
                    # print("결과:", tmpX, tmpY)
                    path_now.append((tmpX, tmpY))
                    x, y = tmpX, tmpY

        path_now.reverse()

        if i != len(points) - 1:
            path_now.pop()

        path += path_now

        path_now = []

        start_point = (desX, desY)

    ####################### Write Your Code Here ################################

    return path

    ############################################################################
