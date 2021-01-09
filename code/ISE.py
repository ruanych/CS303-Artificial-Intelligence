import multiprocessing as mp
import time
import sys
import argparse
import os
import numpy as np

core = 8
N = 1000


def IC_one_core(graph, seed, end_time):
    result = 0
    for i in range(0, N):
        result += IC(graph, seed)
        if time.time() > end_time:
            return result, i + 1
    return result, N


def LT_one_core(graph, seed, end_time):
    result = 0
    for i in range(0, N):
        result += LT(graph, seed)
        if time.time() > end_time:
            return result, i + 1
    return result, N


def IC(graph, seed):
    inactive_flag = [True] * len(graph)
    for s in seed:
        inactive_flag[s] = False

    active_set = seed.copy()
    count = len(active_set)
    while len(active_set) != 0:
        new_active_set = []
        for ac_seed in active_set:
            # neighbor =  [(neighbor,weight), (neighbor,weight), ...]
            neighbor = graph[ac_seed]
            for neig in neighbor:
                if inactive_flag[neig[0]]:
                    probability = np.random.random()
                    if probability < neig[1]:
                        inactive_flag[neig[0]] = False
                        new_active_set.append(neig[0])
        count += len(new_active_set)
        active_set = new_active_set
    return count


def LT(graph, seed):
    active_set = seed.copy()

    graph_size = len(graph)

    w_total = [0.0] * graph_size
    inactive_flag = [True] * graph_size

    for s in seed:
        inactive_flag[s] = False

    thresholds = np.random.uniform(0, 1, graph_size)
    thresholds[0] = 1

    for ac in np.where(thresholds == 0)[0]:
        if inactive_flag[ac]:
            inactive_flag[ac] = False
            active_set.append(ac)


    count = len(active_set)
    while len(active_set) != 0:
        new_active_set = []
        for ac_seed in active_set:
            for acc in graph[ac_seed]:
                w_total[acc[0]] += acc[1]
            # neighbor =  [(neighbor,weight), (neighbor,weight), ...]
            neighbor = graph[ac_seed]
            for neig in neighbor:
                if inactive_flag[neig[0]]:
                    if w_total[neig[0]] >= thresholds[neig[0]]:
                        inactive_flag[neig[0]] = False
                        new_active_set.append(neig[0])
        count += len(new_active_set)
        active_set = new_active_set
    return count


def get_graph(file_name):
    f = open(file_name, 'r')
    graph_raw_str = f.read()
    graph_raw_arr = graph_raw_str.replace('\n', ' ').split(" ")
    it = iter(graph_raw_arr)

    node_count = int(next(it))
    edge_count = int(next(it))
    graph = [[] for _ in range(node_count + 1)]
    for x in range(edge_count):
        i = int(next(it))
        j = int(next(it))
        w = float(next(it))
        graph[i].append((j, w))

    f.close()
    return graph


def get_seed(file_name):
    f = open(file_name, 'r')
    seed_serial = f.read().split('\n')
    seed_serial = [int(x) for x in seed_serial]
    f.close()
    return seed_serial


if __name__ == '__main__':
    '''
    从命令行读参数示例
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='network.txt')
    parser.add_argument('-s', '--seed', type=str, default='seeds.txt')
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=60)

    args = parser.parse_args()
    file_name = args.file_name
    seed = args.seed
    model = args.model
    time_limit = args.time_limit

    end_time = time.time() + time_limit - 2

    seed = get_seed(seed)

    graph = get_graph(file_name)

    '''
    多进程示例
    '''
    # np.random.seed(1)
    pool = mp.Pool(core)
    result = []

    if model == 'IC':
        for i in range(core):
            result.append(pool.apply_async(IC_one_core, args=(graph, seed, end_time)))
    else:
        for i in range(core):
            result.append(pool.apply_async(LT_one_core, args=(graph, seed, end_time)))

    pool.close()
    pool.join()

    count = 0
    total_result = 0
    for re in result:
        t1, t2 = re.get()
        total_result += t1
        count += t2

    print(total_result / count)

    '''
    程序结束后强制退出，跳过垃圾回收时间, 如果没有这个操作会额外需要几秒程序才能完全退出
    '''
    sys.stdout.flush()
