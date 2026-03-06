import numpy as np

class EKF:
    def __init__(self, enc_weight, imu_weight, r, b):
        self.w_enc = enc_weight
        self.w_imu = imu_weight
        self.r = r
        self.b = b