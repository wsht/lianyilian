
#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
total = 31
cur_map = [[1, 1, 1, 1, 1, 1],
           [1, 1, 1, -1, 1, -1],
           [1, 1, -1, 1, 1, -1],
           [1, 1, 1, 1, 1, 1],
           [1, -1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1]]


def point_next(x, y, direction):
    if direction == 0:
        return [x, y-1]
    if direction == 1:
        return [x+1, y]
    if direction == 2:
        return [x, y+1]
    if direction == 3:
        return [x-1, y]


def is_direction_point_valid(point, cur_map):
    max_x = max_y = 5
    if point[0] < 0 or point[0] > max_x or point[1] < 0 or point[1] > max_y or cur_map[point[1]][point[0]] != 1:
        return False
    else:
        return True

def print_path_index(stack):
    for item in stack:
        print item[0][1], item[0][0]


point_start = [3, 0]

stack = []
cur_item = []
cur_item.append(point_start)
cur_item.append(-1)
cur_item.append([])
# exit()

while True:
    next_point = []
    cur_x = cur_item[0][0]
    cur_y = cur_item[0][1]
    # print 'cur point is', cur_item[0]
    for direction in range(0, 4):
        try:
            index  = cur_item[2].index(direction)
            # print 'point:', cur_item[0], direction, 'alread search'
            continue
        except BaseException as identifier:
            next_point = point_next(cur_x, cur_y, direction)
            # print next_point
            if is_direction_point_valid(next_point, cur_map):
                cur_item[1] = direction
                break
            else:
                cur_item[2].append(direction)
    if cur_item[1] == -1:
        # print 'cur stack len is', len(stack)
        if total == len(stack) + 1:
            stack.append(cur_item[:])
            print_path_index(stack)
            break
        else:
            if len(stack) == 0:
                exit('not found the path')
            cur_item = stack.pop()
            # print 'pop point is', cur_item[0]
            cur_item[2].append(cur_item[1])
            cur_item[1] = -1
            cur_map[cur_item[0][1]][cur_item[0][0]] = 1
            continue
    else:
        stack.append(cur_item[:])
        cur_map[cur_y][cur_x] = -1
        cur_item[0] = next_point
        cur_item[1] = -1
        cur_item[2] = []


