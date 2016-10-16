import datetime
import time


def time_function_execution(function_to_execute):
    """
    A decorator that computes the execution time of a function
    """
    def compute_execution_time(*args, **kwargs):
        start_time = time.time()
        result = function_to_execute(*args, **kwargs)
        end_time = time.time()
        computation_time = datetime.timedelta(seconds=end_time - start_time)
        print('I am done!')
        print('Computation lasted: {}'.format(computation_time))
        return result
    return compute_execution_time
