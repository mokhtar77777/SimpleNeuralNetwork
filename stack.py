class Stack:
    def __init__(self, initial_size: int = 0):
        assert initial_size >= 0
        self.initial_size = initial_size
        self.cur_size = self.initial_size

        self.stack = [None] * self.initial_size
        self.cur_index = 0

    def push(self, val):
        if self.cur_index >= self.cur_size:
            if self.initial_size == 0:
                self.stack.append(val)
            else:
                extension = [None] * self.initial_size
                self.stack.extend(extension)

                self.stack[self.cur_index] = val
        else:
            self.stack[self.cur_index] = val
        self.cur_index += 1

    def top(self):
        assert self.cur_index > 0
        return self.stack[self.cur_index - 1]

    def pop(self):
        if self.cur_index > 0:
            self.cur_index = self.cur_index - 1

    def is_empty(self):
        return self.cur_index == 0
