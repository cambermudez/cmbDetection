from torch import nn

class cmbNet(nn.Module):
    def __init__(self):
        super(cmbNet,self).__init__()
        self.features = nn.Sequential(

# Changes to network
# 1. less dropout
# 2. Only conv, more conv, GAP
# 3. Powers of 2 for channels

            nn.Conv3d(in_channels=5,out_channels=100,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Dropout3d(0.5),

            nn.Conv3d(in_channels=100, out_channels=50, kernel_size=(1,5,5), padding=(0,2,2)),
            nn.ReLU(),
            nn.Dropout3d(0.2),

            nn.Conv3d(in_channels=50, out_channels=30, kernel_size=(1, 5, 5), padding=(0,2,2)),
            nn.ReLU(),
            nn.Dropout3d(0.2),

            nn.Flatten(),

            nn.Linear(in_features=3*33*33*30,out_features=20),
            nn.ReLU(),
            nn.Dropout3d(0.2),

            nn.Linear(in_features=20, out_features=25),
            nn.ReLU(),
            nn.Dropout3d(0.2),

            nn.Linear(in_features=25, out_features=1),
            nn.ReLU(),
            nn.Dropout3d(0.2),

            nn.Sigmoid()

        )

    def forward(self,x):
        x = self.features(x)
        return x
