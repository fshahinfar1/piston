# from queue import Queue

# ADD_TASK = 10

# class IsolatedProcessCore:
#     def __init__(self, pipe):
#         self.pipe = pipe
#         self.tasks = queue.Queue()
#         self.cmd_handler = {
#             ADD_TASK: self.add_new_task,
#         }

#     def handler_error(self, cmd):
#         raise RuntimeError('Did not found command handler:', cmd)
    
#     def add_new_task(self, cmd):
#         pass
    
#     def poll(self):
#         if not self.pipe.poll():
#             return
#         cmd = self.pipe.recv()
#         handler = self.cmd_handler.get(cmd[0], self.handler_error)
#         handler(cmd)

# def isolated_worker_main(pipe):
#     running = True
#     obj = IsolatedProcessCore(pipe)

#     while running:

#         obj.poll()

#         if self.tasks.empty():
#             self.busy = False
#             # wait until new task arrives
#             self.task_arrival_promise.wait()
#             continue

#         self.busy = True
#         fn, promise = self.tasks.get()

#         if not promise.mark_under_process():
#             # the work was canceled
#             continue

#         # print('Worker', self.id, ':','got task')
#         error = None
#         try:
#             fn()
#         except Exception as e:
#             error = e
#             print('Worker: saw an exception :-(')
#             print(e)
#         # print('Worker', self.id, ':', fn, error)

#         promise.deliver(error)