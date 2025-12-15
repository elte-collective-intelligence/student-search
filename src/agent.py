import numpy as np


class Agent:
    """Rescuer agent entity in the environment."""

    def __init__(self):
        self.name = ""
        self.collide = True
        self.silent = True
        self.size = 0.025
        self.accel = 3.0
        self.max_speed = 0.3
        self.color = np.array([0.85, 0.35, 0.35])  # Rescuer color
        # State
        self.p_pos = np.zeros(2)  # Position
        self.p_vel = np.zeros(2)  # Velocity
        self.c = np.zeros(2)  # Communication
