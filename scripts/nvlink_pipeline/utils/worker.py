import os
import threading


class Promise:
    def __init__(self):
        self.done = threading.Semaphore(value=0)

    def deliver(self):
        # self.done = True
        self.done.release()
    
    def wait(self):
        # while not self.done:
        #     continue
        self.done.acquire()


class Worker:
    counter = 0

    def __init__(self):
        self.thread = threading.Thread(target=Worker.main, args=(self,))
        self.id = Worker.counter
        Worker.counter += 1

        self.tasks = []
        self.task_arrival_promise = Promise()

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
                # wait until new task arrives
                self.task_arrival_promise.wait()
                continue

            self.busy = True
            fn, promise = self.tasks.pop()
            fn()
            promise.deliver()
    
    def add_task(self, fn, *args, **kwargs):
        promise = Promise()
        wrapped = lambda: fn(*args, **kwargs)
        self.tasks.append((wrapped, promise))

        # wake up the worker if it is waiting for a new task
        self.task_arrival_promise.deliver()

        return promise
    
    def die(self):
        self.running = False
        self.task_arrival_promise.deliver()
 
