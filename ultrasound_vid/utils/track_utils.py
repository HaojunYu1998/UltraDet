import torch
from detectron2.structures import Boxes


class KalmanFilter(object):
    """
    The estimated state are 8-dim:
        cx, cy, w, h, cx_v, cy_v, w_v, h_v
    """
    def __init__(self, bbox):
        """
        Params:
            bbox: Boxes
        """
        device = bbox.tensor.device
        # (4, )
        Z = bbox.tensor.flatten()
        self._std_w_pos = 1. / 20
        self._std_w_vel = 1. / 160
        self.dt = 1.0
        # (cx, cy, w, h, cx_v, cy_v, w_v, h_v)
        self.X = torch.cat([Z, torch.zeros(4).to(device)]).reshape(8, 1)  

        # z = z + dt * z_v, where z in {cx, cy, w, h}
        self.A = torch.eye(8).to(device)
        for i in range(4):
            self.A[i, i+4] = self.dt

        self.H = torch.eye(4, 8).to(device)
        self.P = torch.eye(8).to(device) * 10.0
        G = torch.cat([Z * self._std_w_pos, Z * self._std_w_vel])
        self.Q = torch.diag(torch.square(G)).to(device)
        self.R = torch.eye(4).to(device)

    def predict(self):
        """
        Predict state vector X and variance of uncertainty P (covariance).
        Return:
            vector of predicted state estimate
        """
        # (8,N) = (8,8) x (8,N)
        self.X = torch.mm(self.A, self.X)
        # (8,8) = (8,8) x (8,8) x (8,8) + (8,8)
        self.P = torch.mm(self.A, torch.mm(self.P, self.A.T)) + self.Q
        # (cx, cy, w, h)
        correct_boxes = Boxes(self.X[:4, :].T)
        return correct_boxes

    def correct(self, boxes):
        """
        Correct or update state vector u and variance of uncertainty P (covariance).
        Params:
            bbox: Boxes (N, 4)
        """
        # (4, N)
        Z = boxes.tensor.T
        # (4,4) = (4,8) x (8,8) x (8,4) + (4,4)
        C = torch.mm(torch.mm(self.H, self.P), self.H.T) + self.R
        # (8,4) = (8,8) x (8,4) x (4,4)
        K = torch.mm(torch.mm(self.P, self.H.T), torch.inverse(C))
        # (8,N) = (8,N) + (8,4) x [(4,N) - (4,8) x (8,N)]
        self.X = self.X + torch.mm(K, (Z - torch.mm(self.H, self.X)))
        # (8,8) = (8,8) - (8,4) x (4,8) x (8,8)
        self.P = self.P - torch.mm(torch.mm(K, self.H), self.P)
        
