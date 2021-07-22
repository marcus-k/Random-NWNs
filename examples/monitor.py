#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Class to help measure runtime.
# 
# Author: Marcus Kasdorf
# Date:   May 13, 2021

import time

class Runtime:
    """
    Context manager to measure the runtime of the code in the contained block.

    Parameters
    ----------
    message : str
        Output message to print before printing the elapsed time.

    Attributes
    ----------
    elapsed : float
        Contained the elapsed runtime of the context. This value will be None 
        until the context is closed.

    """
    def __init__(self, message="Time elapsed"):
        self._message = message
        self._start_time = None
        self._end_time = None
        self._elapsed = None
    
    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self._end_time = time.perf_counter()
        self._elapsed = self._end_time - self._start_time
        if self._message:
            if type is None:
                print(f"{self._message}: {self._elapsed}")
            else:
                print(f"(ended abruptly) {self._message}: {self._elapsed}")

    @property
    def elapsed(self):
        return self._elapsed