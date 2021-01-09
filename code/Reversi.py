# try
import numpy as np
import random
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)


# don't change the class name


class AI(object):
    start_time = 0
    no_time = False
    chose = (0, 0)
    r = (120, -50, 48, 42, -77, 8, 6, 5, 4, 3)
    weight = np.array([[r[0], r[1], r[2], r[3], r[3], r[2], r[1], r[0]],
                       [r[1], r[4], r[5], r[6], r[6], r[5], r[4], r[1]],
                       [r[2], r[5], r[7], r[8], r[8], r[7], r[5], r[2]],
                       [r[3], r[6], r[8], r[9], r[9], r[8], r[6], r[3]],
                       [r[3], r[6], r[8], r[9], r[9], r[8], r[6], r[3]],
                       [r[2], r[5], r[7], r[8], r[8], r[7], r[5], r[2]],
                       [r[1], r[4], r[5], r[6], r[6], r[5], r[4], r[1]],
                       [r[0], r[1], r[2], r[3], r[3], r[2], r[1], r[0]]])
    direction = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))

    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []

    # The input is current chessboard.
    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        # ==================================================================
        # Write your algorithm here
        # Here is the simplest sample:Random decision
        # print(chessboard)
        self.start_time = time.time() - 0.2
        self.no_time = False

        none_idx = self.where_are_you(chessboard, COLOR_NONE)

        idxes, my_dir = self.check_all_move(chessboard, self.color, none_idx)

        self.candidate_list = idxes.copy()
        if len(idxes) < 2:
            return self.candidate_list

        none = len(none_idx)

        if none > 53:
            depth = 1
        elif none > 13:
            depth = 2
        else:
            depth = 3

        count = len(idxes)
        if count < 4:
            depth = 3
        elif count < 8:
            depth = 2
        elif count > 9:
            depth = 1

        idxes_dir_score = []
        for i in range(count):
            idxes_dir_score.append([idxes[i], my_dir[i], float('-inf')])

        while True:
            for i in range(count):
                chessboard_new = self.move(chessboard.copy(), idxes_dir_score[i][0], self.color, idxes_dir_score[i][1])
                socre = -self.alpha_beta(chessboard_new, -self.color, depth, float('-inf'), float('inf'))
                if self.no_time:
                    break
                    # return self.candidate_list
                idxes_dir_score[i][2] = socre

            idxes_dir_score = sorted(idxes_dir_score, key=lambda x: x[2], reverse=True)
            self.candidate_list.append(idxes_dir_score[0][0])

            # print(idxes_dir_score)

            if idxes_dir_score[0][2] == float('inf') or depth > 15:
                return self.candidate_list

            depth += 1
            # break

        # ==============Find new pos========================================
        # Make sure that the position of your decision in chess board is empty.
        # If not, the system will return error.
        # Add your decision into candidate_list, Records the chess board
        # You need add all the positions which is valid
        # candidate_list example: [(3,3),(4,4)]
        # You need append your decision at the end of the candidate_list,
        # we will choose the last element of the candidate_list as the position you choose
        # If there is no valid position, you must return a empty list.

    # Check whether the falling position is legal, if so, how many points can we get
    def check_move(self, chessboard, idx, color):
        score = 0
        x, y = idx
        move_dir = []

        for di in range(8):
            move = 1
            # Check for out of bounds and whether it is the opponent's chess piece in the middle
            while (0 <= (x + (move + 1) * self.direction[di][0]) < 8) and (
                    0 <= (y + (move + 1) * self.direction[di][1]) < 8) and (
                    (chessboard[x + move * self.direction[di][0]][y + move * self.direction[di][1]]) == -color):
                # Check the edges for their own pieces
                if color == chessboard[x + (move + 1) * self.direction[di][0]][y + (move + 1) * self.direction[di][1]]:
                    score += move
                    move_dir.append(di)
                    break
                move += 1
        return score, move_dir

    def where_are_you(self, chessboard, color):
        idxes = np.where(chessboard == color)
        return list(zip(idxes[0], idxes[1]))

    def check_all_move(self, chessboard, color, none_idx):
        idxes_ok = []
        move_dir_ok = []
        for idx in none_idx:
            score, move_dir = self.check_move(chessboard, idx, color)
            if score != 0:
                idxes_ok.append(idx)
                move_dir_ok.append(move_dir)
        return idxes_ok, move_dir_ok

    def move(self, chessboard, idx, color, my_dir):
        chessboard[idx] = color
        for di in my_dir:
            x = idx[0] + self.direction[di][0]
            y = idx[1] + self.direction[di][1]

            # Check for out of bounds and whether it is the opponent's chess piece in the middle
            while chessboard[x][y] != color:
                chessboard[x][y] = color
                x = x + self.direction[di][0]
                y = y + self.direction[di][1]
        return chessboard

    def alpha_beta(self, chessboard, color, depth, alpha, beta):
        if time.time() - self.start_time > self.time_out:
            self.no_time = True
            return alpha

        none_idx = self.where_are_you(chessboard, COLOR_NONE)
        my_move, my_dir = self.check_all_move(chessboard, color, none_idx)
        if len(my_move) == 0 and len(self.check_all_move(chessboard, -color, none_idx)[0]) == 0:
            my_piece_count = len(self.where_are_you(chessboard, color))
            your_piece_count = len(self.where_are_you(chessboard, -color))
            if my_piece_count > your_piece_count:
                return float('inf')
            elif my_piece_count < your_piece_count:
                return float('-inf')
            else:
                return 0

        if 0 == depth:
            return self.evaluate(chessboard, color)
        # Generate legal moves

        ii = 0
        for i in my_move:
            chessboard_next = self.move(chessboard.copy(), i, color, my_dir[ii])
            ii = ii + 1
            self.chose = i
            val = -self.alpha_beta(chessboard_next, -color, depth - 1, -beta, -alpha)
            if val >= beta:
                return beta
            if val > alpha:
                alpha = val
        return alpha

    def weight_borad(self, chessboard, color):

        update = 110
        update1 = 55
        update2 = 50
        weight_tmp = self.weight.copy()

        if chessboard[0][0] == color:
            weight_tmp[0][1] = self.r[0]
            weight_tmp[1][1] = update
            weight_tmp[1][0] = self.r[0]

            weight_tmp[0][2] = update1
            weight_tmp[2][0] = update1
            weight_tmp[0][3] = update2
            weight_tmp[3][0] = update2
        if chessboard[0][7] == color:
            weight_tmp[0][6] = self.r[0]
            weight_tmp[1][7] = self.r[0]
            weight_tmp[1][6] = update

            weight_tmp[0][5] = update1
            weight_tmp[2][7] = update1
            weight_tmp[0][4] = update2
            weight_tmp[3][7] = update2
        if chessboard[7][0] == color:
            weight_tmp[7][1] = self.r[0]
            weight_tmp[6][1] = update
            weight_tmp[6][0] = self.r[0]

            weight_tmp[7][2] = update1
            weight_tmp[5][0] = update1
            weight_tmp[7][3] = update2
            weight_tmp[4][0] = update2
        if chessboard[7][7] == color:
            weight_tmp[7][6] = self.r[0]
            weight_tmp[6][6] = update
            weight_tmp[6][7] = self.r[0]

            weight_tmp[7][5] = update1
            weight_tmp[5][7] = update1
            weight_tmp[7][4] = update2
            weight_tmp[4][7] = update2
        value = color * (sum(sum(chessboard * weight_tmp)) + 25 * weight_tmp[self.chose])
        return value

    def stable_count(self, chessboard, color, none_idxes):
        tmp = chessboard.copy()
        for idx in none_idxes:
            for di in range(8):
                x = idx[0] + self.direction[di][0]
                y = idx[1] + self.direction[di][1]
                while (0 < x < 7) and (0 < y < 7):
                    tmp[x][y] = COLOR_NONE
                    x = x + self.direction[di][0]
                    y = y + self.direction[di][1]

        idxes = self.where_are_you(tmp, color)
        return len(idxes)

    def check_edge_none(self, chessboard, ides):
        count = 0
        for my_idx in ides:
            for di in range(8):
                x = my_idx[0] + self.direction[di][0]
                y = my_idx[1] + self.direction[di][1]
                if (0 <= x < 8) and (0 <= y < 8) and chessboard[x][y] == COLOR_NONE:
                    count += 1
        return count

    def evaluate(self, chessboard, color):
        none_p = self.where_are_you(chessboard, COLOR_NONE)
        none = len(none_p)

        if none > 53:
            action_weight = 460
            weight_weight = 48
            stable_weight = 35
            edge_point_weight = 40
        elif none > 13:
            action_weight = 550
            weight_weight = 58
            stable_weight = 66
            edge_point_weight = 60
        else:
            action_weight = 400
            weight_weight = 42
            stable_weight = 88
            edge_point_weight = 80

        value = 0
        if none < 6:
            value += self.stable_count(chessboard, color, none_p) * stable_weight

        value = weight_weight * self.weight_borad(chessboard, color) + \
                action_weight * (len(self.check_all_move(chessboard, color, none_p)[0]) - len(
            self.check_all_move(chessboard, -color, none_p)[0]))

        if none > 10:
            my_p = self.where_are_you(chessboard, color)
            value -= edge_point_weight * self.check_edge_none(chessboard, my_p)
            your_p = self.where_are_you(chessboard, -color)
            value += edge_point_weight * self.check_edge_none(chessboard, your_p)
        # else:
        #     my_set = set()
        #     you_set = set()
        #     for none_idx in none_p:
        #         for di in range(8):
        #             x = none_idx[0] + self.direction[di][0]
        #             y = none_idx[1] + self.direction[di][1]
        #             if (0 <= x < 8) and (0 <= y < 8):
        #                 if chessboard[x][y] == color:
        #                     my_set.add(none_idx)
        #                 elif chessboard[x][y] == -color:
        #                     you_set.add(none_idx)
        #     value += edge_point_weight * (len(you_set) - len(my_set))

        return int(value)
