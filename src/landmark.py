import numpy as np


class Landmark:
    """Landmark entity in the environment (trees or safe zones)."""

    def __init__(self):
        self.name = ""
        self.tree = False
        self.collide = False
        self.movable = False
        self.size = 0.03
        self.boundary = False
        self.type = None  # Safe zone type (A, B, C, D)
        self.color = np.array([0.0, 0.0, 0.0])
        # State
        self.p_pos = np.zeros(2)
        self.p_vel = np.zeros(2)
