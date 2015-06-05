"""
Clone of 2048 game.
"""

import poc_2048_gui
import random

# Directions, DO NOT MODIFY
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

# Offsets for computing tile indices in each direction.
# DO NOT MODIFY this dictionary.
OFFSETS = {UP: (1, 0),
           DOWN: (-1, 0),
           LEFT: (0, 1),
           RIGHT: (0, -1)}

def merge(line):
    """
    Function that merges a single row or column in 2048.
    """
    merged_line = [0] * len(line)
    prev_elem = -1
    merged_idx = 0
    for elem in line:
        if elem == 0:
            continue
        elif elem == prev_elem:
            merged_line[merged_idx - 1] = 2 * elem
            prev_elem = -1
            continue
        else: # elem != prev_elem
            merged_line[merged_idx] = elem
            prev_elem = elem
            merged_idx += 1
        
    return merged_line


class TwentyFortyEight:
    """
    Class to run the game logic.
    """

    def __init__(self, grid_height, grid_width):
        self._height = grid_height
        self._width = grid_width
        self._board = [[0 for _ in range(self._width)] \
                          for _ in range(self._height)]
        self._num_empty = grid_height * grid_width
        self.reset()

    def reset(self):
        """
        Reset the game so the grid is empty except for two
        initial tiles.
        """
        self._board = [[0 for _ in range(self._width)] \
                          for _ in range(self._height)]
        self._num_empty = self._width * self._height
        self.new_tile()
        self.new_tile()

    def __str__(self):
        """
        Return a string representation of the grid for debugging.
        """
        reprstr = self._board.__str__()
        return reprstr

    def get_grid_height(self):
        """
        Get the height of the board.
        """
        return self._height

    def get_grid_width(self):
        """
        Get the width of the board.
        """
        return self._width

    def move(self, direction):
        """
        Move all tiles in the given direction and add
        a new tile if any tiles moved.
        """
        moved = False
        if direction == UP:
            moved = self.move_up()
        
        if direction == DOWN:
            moved = self.move_down()
        
        if direction == LEFT:
            moved = self.move_left()
        
        if direction == RIGHT:
            moved = self.move_right()

        if moved:
            self.new_tile()

    def move_up(self):
        """
        Move tiles up.
        """
        moved = False
        for col in range(self._width):
            line = [self.get_tile(row, col) for row in range(self._height)]
            line_new = merge(line)
            if line != line_new:
                moved = True
            for row in range(self._height):
                self.set_tile(row, col, line_new[row])
        return moved

    def move_down(self):
        """
        Move tiles down.
        """
        moved = False
        for col in range(self._width):
            line = [self.get_tile(row, col) for row in range(self._height-1, -1, -1)]
            line_new = merge(line)
            if line != line_new:
                moved = True
            for row in range(self._height):
                self.set_tile(row, col, line_new[self._height - 1 - row])
        return moved
    
    def move_left(self):
        """
        Move tiles left.
        """
        moved = False
        for row in range(self._height):
            line = [self.get_tile(row, col) for col in range(self._width)]
            line_new = merge(line)
            if line != line_new:
                moved = True
            for col in range(self._width):
                self.set_tile(row, col, line_new[col])
        return moved
    
    def move_right(self):
        """
        Move tiles right.
        """
        moved = False
        for row in range(self._height):
            line = [self.get_tile(row, col) for col in range(self._width-1, -1, -1)]
            line_new = merge(line)
            if line != line_new:
                moved = True
            for col in range(self._width):
                self.set_tile(row, col, line_new[self._width - 1 - col])
        return moved

                
    def new_tile(self):
        """
        Create a new tile in a randomly selected empty
        square.  The tile should be 2 90% of the time and
        4 10% of the time.
        """
        if self._num_empty == 0:
            print "No spot is availabe for new tiles."
            return
        rand_index = random.randrange(0, self._num_empty)
        count = -1
        row_index = 0
        col_index = 0
        while True: 
            if self._board[row_index][col_index] == 0:
                count += 1
            if count == rand_index:
                break
            col_index += 1
            if col_index == self._width:
                col_index = 0
                row_index += 1
            #print rand_index, count, row_index, col_index
        rand_num = random.random()
        value = 2 if rand_num < 0.9 else 4
        self.set_tile(row_index, col_index, value)        
        

    def set_tile(self, row, col, value):
        """
        Set the tile at position row, col to have the given value.
        """
        if self._board[row][col] == 0 and value != 0:
            self._num_empty -= 1
        elif self._board[row][col] != 0 and value == 0:
            self._num_empty += 1
        self._board[row][col] = value

    def get_tile(self, row, col):
        """
        Return the value of the tile at position row, col.
        """
        return self._board[row][col]

poc_2048_gui.run_gui(TwentyFortyEight(4, 4))
