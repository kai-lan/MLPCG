'''
File: benchmark.py
File Created: Monday, 17th April 2023 12:17:40 am

Author: Kai Lan (kai.weixian.lan@gmail.com)
--------------
'''
import time
import sys


class MyTimer:
    def __init__(self, fun_names):
        self.start_time = {}
        self.total_time = {}
        self.counter = {}
        for name in fun_names:
            self.start_time[name] = 0.0
            self.total_time[name] = 0.0
            self.counter[name] = 0
        def tracefunc(frame, event, arg):
            fun_name = frame.f_code.co_name
            if event == "call":
                if fun_name in self.counter.keys():
                    self.counter[fun_name] += 1
                    self.start_time[fun_name] = time.perf_counter()
            elif event == "return":
                if fun_name in self.counter.keys():
                    self.total_time[fun_name] += time.perf_counter() - self.start_time[fun_name]
            return tracefunc
        sys.setprofile(tracefunc)

    def reset(self):
        for fun_name in self.start_time.keys():
            self.start_time[fun_name] = 0.0
            self.total_time[fun_name] = 0.0
            self.counter[fun_name] = 0


if __name__ == '__main__':
    def say_whee(i, j):
        time.sleep(1)
        print("Whee! ", i, j)
        return i + j


    def foo(x):
        time.sleep(0.5)
        print("foo")
        return x

    timer = MyTimer(['foo', 'say_whee'])
    # print(timer.timers)
    for _ in range(1):
        foo(11)
    # for _ in range(2):
    #     say_whee(1, 2)
    # time.sleep(1)
    # foo(11)
    # foo(11)

    timer.reset()
    foo(11)
    print('-'*30)
    print(timer.counter)
    print(timer.total_time)
