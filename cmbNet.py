from torch import nn

class cmbNet(nn.Module):
    def __init__(self):
        super(cmbNet,self).__init__()
        self.features = nn.Sequential(

            nn.Conv3d(in_channels=5,out_channels=100,kernel_size=3,padding=0),
            nn.ReLU(),
            nn.Dropout3d(0.5),

            nn.Conv3d(in_channels=100, out_channels=50, kernel_size=(5,5,1), padding=0),
            nn.ReLU(),
            nn.Dropout3d(0.2),

            nn.Conv3d(in_channels=50, out_channels=30, kernel_size=(5, 5, 1), padding=0),
            nn.ReLU(),
            nn.Dropout3d(0.2),

            nn.Flatten(),

            nn.Linear(in_channels=23*23*30,out_channels=20),
            nn.Relu(),
            nn.Dropout3d(0.2),

            nn.Linear(in_channels=20, out_channels=25),
            nn.Relu(),
            nn.Dropout3d(0.2),

            nn.Linear(in_channels=25, out_channels=1),
            nn.Relu(),
            nn.Dropout3d(0.2),

            nn.Softmax(dim=1)

        )

    def forward(self,x):
        x = self.features(x)
        return x
