import multiprocessing as mp
import time
import sys
import argparse
import os
import numpy as np
import random


core = 8


def IC(graph: list, seed: int):
    seed_list = []

    inactive_flag = [True] * len(graph)
    inactive_flag[seed] = False
    active_set = [seed]
    while len(active_set) != 0:
        seed_list.extend(active_set)
        new_active_set = []
        for ac_seed in active_set:
            # neighbor =  [(neighbor,weight), (neighbor,weight), ...]
            neighbor = graph[ac_seed]
            for neig in neighbor:
                if inactive_flag[neig[0]]:
                    # probability = np.random.random()
                    probability = random.random()
                    # probability = random.uniform(0.0, 1.0)
                    if probability < neig[1]:
                        inactive_flag[neig[0]] = False
                        new_active_set.append(neig[0])

        active_set = new_active_set
    return seed_list


def get_IC_RRsets_one_core(graph, g_size, end_time, max_n):
    R = []
    inital = 160000
    np.random.seed(time.time_ns() % 999983)
    random_list = np.random.randint(1, g_size, size=inital)
    # random_list = [random.randint(1, g_size - 1) for _ in range(2000000)]
    for ran in random_list:
        # IC(graph, seed) RRset list [1,2,...]
        R.append(IC(graph, ran))
    R_count = inital
    while True:
        random_list = np.random.randint(1, g_size, size=100000)
        # random_list = [random.randint(1, g_size - 1) for _ in range(2000000)]
        for ran in random_list:
            # IC(graph, seed) RRset list [1,2,...]
            R.append(IC(graph, ran))
            R_count += 1
            if time.time() > end_time or R_count > max_n:  # MB
                return R


def get_LT_RRsets_one_core(graph, g_size, end_time, max_n):
    R = []
    inital = 280000
    np.random.seed(time.time_ns() % 999983)
    random_list = np.random.randint(1, g_size, size=inital)
    for seed in random_list:
        RRset = set()
        current_node = seed
        while True:
            RRset.add(current_node)
            neig_count = len(graph[current_node])
            if neig_count == 0:
                break

            node_idx = random.randint(0, neig_count - 1)

            current_node = graph[current_node][node_idx][0]
            if current_node in RRset:
                break
        R.append(list(RRset))
    R_count = inital
    while True:
        random_list = np.random.randint(1, g_size, size=100000)
        for seed in random_list:
            RRset = set()
            current_node = seed
            while True:
                RRset.add(current_node)
                neig_count = len(graph[current_node])
                if neig_count == 0:
                    break
                # p = np.random.random()
                node_idx = random.randint(0, neig_count - 1)
                # p = random.uniform(0.0, 1.0)
                # w = graph[current_node][0][1]

                current_node = graph[current_node][node_idx][0]
                if current_node in RRset:
                    break
            R.append(list(RRset))
            R_count += 1
            if time.time() > end_time or R_count > max_n:
                return R


def get_reverse_graph(file_name):
    f = open(file_name, 'r')
    graph_raw_str = f.read()
    graph_raw_arr = graph_raw_str.replace('\n', ' ').split(" ")
    it = iter(graph_raw_arr)

    node_count = int(next(it))
    edge_count = int(next(it))
    graph = [[] for _ in range(node_count + 1)]
    _in_degree = [0] * (node_count + 1)
    for x in range(edge_count):
        j = int(next(it))
        i = int(next(it))
        w = float(next(it))
        _in_degree[j] += 1
        graph[i].append((j, w))

    f.close()
    return graph, _in_degree


if __name__ == '__main__':
    '''
    从命令行读参数示例
    '''
    test_start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='DatasetOnTestPlatform/NetHEPT.txt')
    parser.add_argument('-k', '--seed_size', type=int, default=500)
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=120)

    args = parser.parse_args()
    file_name = args.file_name
    k = args.seed_size
    model = args.model
    time_limit = args.time_limit

    reverse_graph, in_degree = get_reverse_graph(file_name)

    g_size = len(in_degree)


    if time_limit < 61:
        get_rrset_time = 26
        max_n = 360000
    else:
        get_rrset_time = time_limit // 2 + 6
        max_n = 4760 * time_limit

    if g_size > 30000:
        get_rrset_time = time_limit // 3
    elif g_size > 100000:
        get_rrset_time = time_limit // 3 - 5
    elif g_size > 300000:
        get_rrset_time = time_limit // 4 - 3
    elif g_size > 500000:
        get_rrset_time = time_limit // 4 - 8

    end_time = time.time() + get_rrset_time


    pool = mp.Pool(core)
    result = []
    if model == 'IC':
        for i in range(core):
            result.append(pool.apply_async(get_IC_RRsets_one_core, args=(reverse_graph, g_size, end_time, max_n)))
    else:
        end_time = end_time - 3
        for i in range(core):
            result.append(pool.apply_async(get_LT_RRsets_one_core, args=(reverse_graph, g_size, end_time, max_n)))

    pool.close()
    pool.join()

    big_R = []
    big_dict = {}
    for re in result:
        subR = re.get()
        big_R.extend(subR)

    for i, RRset in enumerate(big_R):
        for node in RRset:
            if node in big_dict:
                big_dict[node][0] += 1
                big_dict[node][1].append(i)
            else:
                big_dict[node] = [1, [i]]

    seed_set = set()
    # RRset removed from big_R
    RRset_flag = [True] * len(big_R)

    while len(seed_set) < k:
        seed_key, seed_value = max(big_dict.items(), key=lambda x: x[1][0])

        if big_dict[seed_key][0] < 0:
            break
        seed_set.add(seed_key)
        seed_value[0] = -666
        for node_rrset in seed_value[1]:
            # if not removed
            if RRset_flag[node_rrset]:
                RRset_flag[node_rrset] = False
                for rrset_node in big_R[node_rrset]:
                    big_dict[rrset_node][0] -= 1

    if len(seed_set) < k:
        in_degree_idx = np.argsort(-np.array(in_degree))
        idx_it = iter(in_degree_idx)
        while len(seed_set) < k:
            seed_set.add(int(next(idx_it)))

    for i in seed_set:
        print(i)

    '''
    程序结束后强制退出，跳过垃圾回收时间, 如果没有这个操作会额外需要几秒程序才能完全退出
    '''
    sys.stdout.flush()
    sys.exit(0)
