import time

class SingleClock:
    def __init__(self, name, indent=0):
        self.name = name
        self.indent = indent
        self.tot_time = 0.0
        self.children = {}
    def start_timer(self):
        self.start = time.perf_counter()
    def stop_timer(self):
        self.tot_time += time.perf_counter() - self.start
    def info(self):
        info = 2*' '*self.indent + self.name + ': ' + str(self.tot_time) + '\n'
        for name, clock in self.children.items():
            info += clock.info()
        return info

class GlobalClock:
    def __init__(self):
        self.top_level_clocks = {}
        self.scope = []
    def start(self, name):
        clock = None
        if len(self.scope) == 0:
            if not name in self.top_level_clocks.keys():
                self.top_level_clocks[name] = SingleClock(name)
            clock = self.top_level_clocks[name]
        else:
            levels = len(self.scope)
            clock = self.scope[0]
            for i in range(1, levels):
                clock = clock.children[self.scope[i].name]
            if not name in clock.children.keys():
                clock.children[name] = SingleClock(name, indent=levels)
            clock = clock.children[name]
        clock.start_timer()
        self.scope.append(clock)


    def stop(self, name):
        if len(self.scope) == 0:
            if name in self.top_level_clocks.keys():
                self.top_level_clocks[name].stop_timer()
            else: raise Exception(f"Clock '{name}' was not initialized.")
        else:
            if name == self.scope[-1].name:
                self.scope.pop().stop_timer()
            else: raise Exception("Timers must be stopped in order.")

    def report(self):
        for name, clock in self.top_level_clocks.items():
            print(clock.info())


if __name__ == '__main__':

    clock = GlobalClock()

    clock.start('a')
    clock.start('aa')
    time.sleep(2)
    clock.stop('aa')
    clock.start('b')
    time.sleep(3)
    clock.stop('b')
    clock.stop('a')
    clock.report()

