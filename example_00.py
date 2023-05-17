import math
import datetime
import multiprocessing as mp
import threading as thd


def train_on_parameter(name, param):
    print("... {} is running".format(name))
    result = 0
    for num in param:
        result += math.sqrt(num * math.tanh(num) / math.log2(num) / math.log10(num))
    return {name: result}


def compare(t_param_dict):
    start_t = datetime.datetime.now()
    to_join_list = []
    for name, param in t_param_dict.items():
        p = mp.Process(target=train_on_parameter, args=(name, param))
        p.start()
        to_join_list.append(p)
    for p in to_join_list:
        p.join()
    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print("多进程计算 共消耗: " + "{:.2f}".format(elapsed_sec) + " 秒")

    print("-" * 120)

    start_t = datetime.datetime.now()
    to_join_list = []
    for name, param in t_param_dict.items():
        t = thd.Thread(target=train_on_parameter, args=(name, param))
        t.start()
        to_join_list.append(t)
    for t in to_join_list:
        t.join()
    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print("多线程计算 共消耗: " + "{:.2f}".format(elapsed_sec) + " 秒")
    return


if __name__ == "__main__":
    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    param_dict = {"task1": list(range(10, 3000000)),
                  "task2": list(range(3000000, 6000000)),
                  "task3": list(range(6000000, 9000000)),
                  "task4": list(range(9000000, 12000000)),
                  "task5": list(range(12000000, 15000000)),
                  "task6": list(range(15000000, 18000000)),
                  "task7": list(range(18000000, 21000000)),
                  "task8": list(range(21000000, 24000000))}

    compare(t_param_dict=param_dict)
