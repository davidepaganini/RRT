
class TemporalTreeNode:
    def __init__(self, joints):
        self.joints = joints
        self.parent = None
        self.cost = 0.0