#!/usr/bin/python
# -*- coding: UTF-8 -*-
import wda
import cv2
import numpy as np
import time
import sys
wda.DEBUG = False
wda.HTTP_TIMEOUT = 60.0

class LianLianKan:
    #wda client
    wda = 0
    session = 0
    img_rgb = 0
    img_gray = 0
    #需要查找的图片
    search = 0
    #需要祛除的头部图片
    head = 0
    #需要去除的尾部图片
    bottom = 0

    img_rgb_size = [0, 0]
    #不同级别，每个格的边长是不一样的
    slide_width_list = [149, 129, 123]
    #不同级别，每个格的空隙是不一样的
    border_width_list = [39, 34, 19]

    cur_slide_width = 149
    cur_border_width = 39

    valid_area_start_point = [0, 0]
    valid_area_end_point = [0, 0]

    top_start = 398
    screenshot_png = 'screenshot.png'

    # template_start_png = 'search2.png'
    # template_normal_png = 'search.png'

    start_point = []
    end_point = []

    lian_map = []
    total = 0
    #当前关卡数
    cur_gate = 0

    def __init__(self):
        self.wda = wda.Client('http://localhost:8100')
        self.session = self.wda.session()
        self.search = cv2.imread('search.png', 0)
        self.head = cv2.imread('head.png', 0)
        self.bottom = cv2.imread('bottom.png', 0)

    def run(self, cur_gate):
        self.cur_gate = cur_gate
        while True:
            self.screenshot()
            self.handle_screenshot()
            print self.lian_map
            path = self.find_path()
            print path
            real = self.back_to_real_path(path)
            self.move(real)
            self.clear()
            self.next()

    def next(self):
        if self.cur_gate % 5 == 0:
            time.sleep(1)
            self.session.tap(355, 250)
        time.sleep(0.5)
        self.session.tap(172, 606)
        self.cur_gate += 1
        time.sleep(1)

    def test_next(self, cur_gate):
        self.session.tap(355, 250)
        time.sleep(0.5)
        self.session.tap(172, 606)
        cur_gate += 1
        time.sleep(1)
        self.run(cur_gate)

    def clear(self):
        self.valid_area_start_point = [0, 0]
        self.valid_area_end_point = [0, 0]
        self.lian_map = []
        self.start_point = []
        self.total = 0
        self.img_rgb_size = [0, 0]

    def screenshot(self):
        self.wda.screenshot(self.screenshot_png)
        self.img_rgb = cv2.imread(self.screenshot_png)
        self.img_gray = cv2.cvtColor(self.img_rgb, cv2.COLOR_BGR2GRAY)
        w, h = self.img_gray.shape[::-1]
        self.img_rgb_size = [w, h]

    def handle_screenshot(self):
        self.pre_handle_screenshot()
        try:
            index = self.slide_width_list.index(self.cur_slide_width)
        except expression as identifier:
            exit('index not found')

        threshold = 0.7
        loc = [[], []]
        for i in range(index, len(self.slide_width_list)):
            self.cur_slide_width = self.slide_width_list[i]
            self.cur_border_width = self.border_width_list[i]
            self.search = cv2.resize(
                self.search, (self.cur_slide_width, self.cur_slide_width))

            w, h = self.search.shape[::-1]
            w = h = self.cur_slide_width
            res = cv2.matchTemplate(
                self.img_gray, self.search, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            if len(loc[0]) > 0:
                break
        if len(loc[0]) < 0:
            exit('not fount')
        self.first_build_map(loc)

    def pre_handle_screenshot(self):
        # 去除head 以及bottom 找到中心区域
        threshold = 0.8
        loc = [[], []]
        res = cv2.matchTemplate(
            self.img_gray, self.bottom, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        end_y, end_x = loc[0][0], loc[0][1]
        self.img_rgb = self.img_rgb[self.top_start:end_y, 0:]
        cv2.imwrite('without_head_bottom.png', self.img_rgb)
        self.img_gray = cv2.cvtColor(self.img_rgb, cv2.COLOR_BGR2GRAY)

    def first_build_map(self, loc):
        cur_y = 0
        cur_x_list = []
        # 记录x坐标点位值
        cur_map_list = []
        row = 0
        max_column = 0
        min_y = max_y = min_x = max_x = 0
        # 这里可以用坐标值计算当前点在那个位置
        for pt in zip(*loc[::-1]):
            cv2.rectangle(self.img_rgb, pt, (pt[0] + self.cur_slide_width,
                                             pt[1] + self.cur_slide_width), (204, 204, 204), 2)
            
            max_y = pt[1]
            if not self.in_range(cur_y, pt[1]):
                if cur_y == 0:
                    min_y = pt[1]
                    min_x = max_x = pt[0]
                else:
                    min_x, max_x = self.second_build_map(
                        cur_map_list, row, cur_x_list, min_x, max_x)
                    row = row+1
                cur_x_list = []
                cur_y = pt[1]

            ret = reduce(lambda accum_value, x: accum_value and not self.in_range(
                x, pt[0]), cur_x_list, True)
            if ret:
                cur_x_list.append(pt[0])
        min_x, max_x = self.second_build_map(
            cur_map_list, row, cur_x_list, min_x, max_x)

        self.valid_area_start_point = [min_x, min_y]
        self.valid_area_end_point = [max_x, max_y]
        cv2.imwrite('res.png', self.img_rgb)
        self.build_map(cur_map_list)

    def find_path(self):
        stack = []
        cur_item = []
        cur_item.append(self.start_point)
        cur_item.append(-1)
        cur_item.append([])
        while True:
            next_point = []
            cur_x = cur_item[0][0]
            cur_y = cur_item[0][1]
            for direction in range(0, 4):
                try:
                    cur_item[2].index(direction)
                    continue
                except BaseException as identifier:
                    next_point = self.point_next(cur_x, cur_y, direction)
                    if self.is_direction_point_valid(next_point):
                        cur_item[1] = direction
                        break
                    else:
                        cur_item[2].append(direction)
            if cur_item[1] == -1:
                if self.total == len(stack) + 1:
                    stack.append(cur_item[:])
                    break
                else:
                    if len(stack) == 0:
                        exit('not found the path')
                    cur_item = stack.pop()
                    cur_item[2].append(cur_item[1])
                    cur_item[1] = -1
                    self.lian_map[cur_item[0][1]][cur_item[0][0]] = 1
                    continue
            else:
                stack.append(cur_item[:])
                self.lian_map[cur_y][cur_x] = -1
                cur_item[0] = next_point
                cur_item[1] = -1
                cur_item[2] = []
        result = []
        for item in stack:
            result.append(item[0])
        return result

    def point_next(self, x, y, direction):
        if direction == 0:
            return [x, y-1]
        if direction == 1:
            return [x+1, y]
        if direction == 2:
            return [x, y+1]
        if direction == 3:
            return [x-1, y]

    def is_direction_point_valid(self, point):
        max_y = len(self.lian_map) - 1
        max_x = len(self.lian_map[0])-1
        if point[0] < 0 or point[0] > max_x or point[1] < 0 or point[1] > max_y or self.lian_map[point[1]][point[0]] != 1:
            return False
        else:
            return True

    def build_map(self, cur_map_list):
        print 'cur map list', cur_map_list
        start_position = self.get_start_position()
        self.rebuild_valid_area_with_start_position(start_position)
        print self.valid_area_start_point, self.valid_area_end_point, self.cur_slide_width, self.cur_border_width
        last_point = [self.valid_area_end_point[0]-self.valid_area_start_point[0],
                      self.valid_area_end_point[1]-self.valid_area_start_point[1]]
        last_index = self.get_index(last_point)
        print 'last index:', last_index
        print 'varlid area:', self.valid_area_start_point, self.valid_area_end_point
        self.lian_map = []
        for i in range(0, last_index[1]+1):
            self.lian_map.append([])
            for j in range(0, last_index[0]+1):
                self.lian_map[i].append(-1)

        for i in range(0, len(cur_map_list)):
            for j in range(0, len(cur_map_list[i])):
                x = self.cal_index(
                    cur_map_list[i][j] - self.valid_area_start_point[0])
                self.lian_map[i][x] = 1
        # 查找起始点坐标
        print 'start position', start_position
        self.start_point = self.get_index(
            [start_position[0] - self.valid_area_start_point[0], start_position[1]-self.valid_area_start_point[1]])
        print 'start point:', self.start_point
        # 这里看是否扫描到起始点
        if self.lian_map[self.start_point[1]][self.start_point[0]] != 1:
            self.total += 1
            self.lian_map[self.start_point[1]][self.start_point[0]] = 1
        print 'start point', self.start_point
        # 检验start point 是否已经在map中 如果不在，总体数量加一 并且设置点值


    def rebuild_valid_area_with_start_position(self, start_position):
        x, y = start_position
        is_in_area = True
        between = self.cur_border_width + self.cur_slide_width
        if x < self.valid_area_start_point[0]:
            self.valid_area_start_point[0] -= between
            is_in_area = False
        if x > self.valid_area_end_point[0] + self.cur_slide_width:
            self.valid_area_end_point[0] += between
            is_in_area = False

        if y < self.valid_area_start_point[1]:
            self.valid_area_start_point[1] -= between
            is_in_area = False
        if y > self.valid_area_end_point[1] + self.cur_slide_width:
            self.valid_area_start_point[1] += between
            is_in_area = False

    def get_start_position(self):
        # return [x,y]
        return self.get_start_position_by_white()

    def get_start_position_by_white(self):
        mask = cv2.inRange(self.img_rgb, (0, 0, 0), (204, 204, 204))
        self.img_rgb[mask != 0] = [204, 204, 204]
        cv2.imwrite('mask.png', mask)
        self.img_gray = cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2GRAY)
        copy_img = self.img_gray
        (minval, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(copy_img)
        search_loc = []
        if self.in_range(204, minval):
            search_loc = maxLoc
        else:
            search_loc = minLoc
        print minval, maxVal, minLoc, maxLoc, 'select', search_loc
        cv2.circle(copy_img, search_loc, 5, (255, 0, 0), 2)
        cv2.imwrite('cropimg.png', copy_img)
        return search_loc


    def get_access_path_num(self, row, column):
        max_row = len(self.lian_map)-1
        max_column = len(self.lian_map[0])-1

        def is_access(
            i, j): return 0 if i < 0 or i > max_row or j < 0 or j > max_column or self.lian_map[i][j] == -1 else 1

        return is_access(row-1, column) + is_access(row+1, column) + is_access(row, column-1) + is_access(row, column+1)



    def in_range(self, base, value):
        return base - 5 <= value and base + 5 >= value

    def back_to_real_path(self, path):
        screen_size = self.session.window_size()
        print screen_size
        print self.img_rgb_size
        width_rate = float(screen_size[0]) / float(self.img_rgb_size[0])
        height_rate = float(screen_size[1])/float(self.img_rgb_size[1])
        print width_rate
        print height_rate
        real = []
        between = self.cur_border_width + self.cur_slide_width
        for item in path:
            tmp = [0, 0]
            tmp[0] = (0 + self.valid_area_start_point[0] +
                      item[0] * between + self.cur_slide_width / 2) * width_rate
            tmp[1] = (self.top_start + self.valid_area_start_point[1] +
                      item[1] * between + self.cur_slide_width / 2) * height_rate
            real.append(tmp)
        print real
        return real

    def move(self, path):
        for item in path:
            self.session.tap(item[0], item[1])

    def reset(self):
        self.start_point = []

    def set_end_point(self):
        return []

    def get_move_path(self):
        return []

    def second_build_map(self, path_map, row, path_list, min_x, max_x):
        self.total += len(path_list)
        path_map.append([])
        path_list.sort()
        path_map[row] = path_list
        if len(path_list):
            min_x = min(path_list[0], min_x)
            max_x = max(path_list[len(path_list)-1], max_x)

        return min_x, max_x

    def get_index(self, point):
        return [self.cal_index(point[0]), self.cal_index(point[1])]

    def cal_index(self, width):

        max_v = width / self.cur_slide_width
        min_v = width / (self.cur_slide_width + self.cur_border_width)
        return min_v
        if max_v == min_v:
            return max_v
        if width < (min_v+1)*self.cur_slide_width + min_v * self.cur_border_width:
            return min_v

        return max_v


test = LianLianKan()
test.run(int(sys.argv[1]))
# test.test_next(int(sys.argv[1]))
# test.screenshot()
exit()
