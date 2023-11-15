from time import sleep
import threading
import queue

class RequestQueue():
    '''
    @param function_to_call: function - function to call for each item in the queue
    @param iterate_args: list - put on a queue and pass each item as 1st param to function_to_call
    @param fixed_args: list - always pass this parameters to function_to_call
    @param call_rate: how often it will remove items from the queue and pass it to function_to_call (in minutes)
    '''
    def __init__(self, function_to_call, iterate_args, fixed_args, call_rate):
        self.data_queue = queue.Queue()
        self.function = function_to_call
        self.fixed_args = fixed_args
        self.request_rate = call_rate
        self.delay = 60/self.request_rate

        # add data to queue
        [self.data_queue.put(item) for item in iterate_args]
    
    def take_a_nap(self):
        sleep(self.delay)
    
    def run(self):
        while not self.data_queue.empty():
            thread = threading.Thread(target=self.function, args=(self.data_queue.get(), *self.fixed_args))
            thread.daemon = True
            thread.start()
            self.take_a_nap()
     
    def print_queue(self):
        while not self.data_queue.empty():
            data = self.data_queue.get()
            self.take_a_nap()