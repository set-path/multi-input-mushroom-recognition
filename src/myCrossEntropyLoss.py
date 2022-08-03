import torch


class myCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(myCrossEntropyLoss, self).__init__()

    def forward(self, y_pred, y_truth):
        P_i = torch.nn.functional.softmax(y_pred, dim=1)
        loss = y_truth*torch.log(P_i + 0.0000001)
        loss = -torch.mean(torch.sum(loss, dim=1), dim=0)
        return loss
