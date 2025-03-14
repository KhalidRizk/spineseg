from monai.networks.nets import SegResNet
import torch.nn as nn


class MonaiUNetR(nn.Module):
    def __init__(self, input_size=(96,96,128)):
        super(MonaiUNetR, self).__init__()
        self.model = SegResNet(
    blocks_down=[1, 2, 2, 4],      # Number of residual blocks at each downsampling level
    blocks_up=[1, 1, 1],           # Number of residual blocks at each upsampling level
    init_filters=16,               # Initial number of filters
    in_channels=1,                 # Input channels (e.g., for CT scans)
    out_channels=2,                # Output classes
    dropout_prob=0.2,              # Dropout probability
    spatial_dims=3                 # 2D or 3D implementation (2 for 2D images)
)

    def forward(self, x):
        logits = self.model(x)
        return logits
    

if __name__ == "__main__":
    from torchinfo import summary
    model = MonaiUNetR((96,96,128))
    # print(model)
    summary(model, input_size=(1,1,96,96,128))