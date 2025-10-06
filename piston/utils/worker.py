import os
import threading
import queue


class Promise:
    def __init__(self):
        self.done = threading.Semaphore(value=0)
        self.error = None

        self._lock = threading.Semaphore(value=1)
        self._under_process = False
        self._cancel = False

    def deliver(self, error=None):
        # self.done = True
        self.done.release()
        self.error = error
    
    def wait(self):
        # while not self.done:
        #     continue
        self.done.acquire()
        if self.error is not None:
            raise self.error
    
    def mark_under_process(self) -> bool:
        if not self._cancel:
            with self._lock:
                if not self._cancel:
                    self._under_process = True
                    return True
        return False
    
    def cancel(self) -> bool:
        if not self._under_process:
            with self._lock:
                if not self._under_process:
                    self._cancel = True
                    return True
        return False


class Worker:
    """
    An abstraction on top of a thread that receives tasks in the form of
    functions and arguments and executes them.
    """
    counter = 0

    def __init__(self):
        self.thread = threading.Thread(target=Worker.main, args=(self,))
        self.id = Worker.counter
        Worker.counter += 1

        self.tasks = queue.Queue()
        self.task_arrival_promise = Promise()

        self.busy = False
        self.running = False

        self.thread.start()
    
    def main(self):
        """
        Main worker loop
        """
        self.running = True
        while self.running:
            # busy poll
            if self.tasks.empty():
                self.busy = False
                # wait until new task arrives
                self.task_arrival_promise.wait()
                continue

            self.busy = True
            fn, promise = self.tasks.get()

            if not promise.mark_under_process():
                # the work was canceled
                continue

            # print('Worker', self.id, ':','got task')
            error = None
            try:
                fn()
            except Exception as e:
                error = e
                print('Worker: saw an exception :-(')
                print(e)
            # print('Worker', self.id, ':', fn, error)

            promise.deliver(error)
    
    def add_task(self, fn, *args, **kwargs):
        """
        Add a task to queue. Worker will invoke function `fn` with given
        arguments.
        Returns a promise to check if the task is finished or not.
        """
        promise = Promise()
        wrapped = lambda: fn(*args, **kwargs)
        self.tasks.put((wrapped, promise))

        # wake up the worker if it is waiting for a new task
        self.task_arrival_promise.deliver()

        return promise
    
    def die(self):
        """
        Stop the worker
        """
        self.running = False
        # interrupt the worker if its blocked on task arrival
        self.task_arrival_promise.deliver()
