


class Tiles():

    def __init__(self, top, left, side,):
        self.cleared = False

        self.top = top
        self.left = left
        self.side = side

        self.center = [left+side//2, top+side//2]





def create_tiles(length, count):
    fog_tiles = []

    for i in range(0, 1000, length//count):
        for j in range(0, 1000, length//count):
            fog_tiles.append(Tiles(j, i, length//count))

    return fog_tiles
