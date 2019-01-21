import random
from collections import deque


class Memory(object):
    def __init__(self):
        self.memory = deque()
