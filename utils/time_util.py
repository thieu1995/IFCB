#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:21, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from functools import wraps
from logging import Logger
from time import process_time, time


def timing(original_function=None, logger=None):
    def _decorate(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            start_time = time()
            start_process_time = process_time()
            try:
                return func(*args, **kwargs)
            finally:
                end_process_time = process_time()
                end_time = time()
                duration = (end_time - start_time) * 1000
                process_duration = (end_process_time - start_process_time) * 1000
                content_1 = f'Func: `{func.__name__}` with args: [{args!r}, {kwargs!r}]'
                content_2 = f'took: {duration:.3f} ms, process time: {process_duration:.3f}'
                content = f'{content_1} {content_2}'
                if isinstance(logger, Logger):
                    logger.info(content)
                else:
                    print(content)

        return wrapped_function

    if original_function:
        return _decorate(original_function)

    return _decorate

