class ResultDataPoint(object):
    def __init__(self, input_image):
        self.input_image = input_image
        self.depth = None
        self.rotation = None
        self.translation = None

    def apply_model(self, model):
        depth, pose = model(self.input_image)
        self.depth = depth
        self.rotation = pose[0]
        self.translation = pose[1]
        return self
