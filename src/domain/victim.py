from enum import Enum
import numpy as np


class VictimState(Enum):
    """States for victim entities."""

    IDLE = 0  # Waiting to be found
    FOLLOW = 1  # Following a rescuer
    STOP = 2  # Saved at safe zone


class Victim:
    """Victim entity in the environment (not an agent - environmental entity)."""

    def __init__(self):
        self.name = ""
        self.collide = True
        self.size = 0.015
        self.speed = 0.2  # Following speed
        self.type = None  # Victim type (A, B, C, D)
        self.color = np.array([0.0, 0.0, 0.0])
        # State
        self.state = VictimState.IDLE  # Victim state (idle, follow, stop)
        self.following_agent = None  # Which agent the victim is following
        self.action_u = None
        self.p_pos = np.zeros(2)  # Position
        self.p_vel = np.zeros(2)  # Velocity
        self.c = np.zeros(2)  # Communication

    @property
    def is_saved(self):
        """Check if the victim has been saved (reached safe zone)."""
        return self.state is VictimState.STOP
