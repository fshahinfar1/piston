import os
import threading


class Promise:
    def __init__(self):
        self.done = False
    
    def wait(self):
        while not self.done:
            continue


class Worker:
    counter = 0

    def __init__(self):
        self.thread = threading.Thread(target=Worker.main, args=(self,))
        self.tasks = []
        self.id = Worker.counter
        Worker.counter += 1
        self.busy = False
        self.running = False

        self.thread.start()
    
    def main(self):
        tid = threading.get_native_id()
        os.sched_setaffinity(tid, [self.id,])

        self.running = True
        while self.running:
            # busy poll
            if not self.tasks:
                self.busy = False
                continue

            self.busy = True
            fn, promise = self.tasks.pop()
            fn()
            promise.done = True
    
    def add_task(self, fn, *args, **kwargs):
        promise = Promise()
        wrapped = lambda: fn(*args, **kwargs)
        self.tasks.append((wrapped, promise))
        return promise
    
    def die(self):
        self.running = False
            