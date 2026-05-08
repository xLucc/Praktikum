class CalibrationError(Exception):
    pass

class InvalidRobotTransform(CalibrationError):
    pass

class FalseIndexing(CalibrationError):
    pass
