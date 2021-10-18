from torch import nn

class cmbNet(nn.Module):
    def __init__(self):
        super(cmbNet,self).__init__()
        self.features = nn.Sequential(

            nn.Conv3d(in_channels=5,out_channels=16,kernel_size=3,padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1),
            nn.LeakyReLU())

        self.pool = nn.AvgPool3d((3,4,4))

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=2048,out_features=512),
            nn.LeakyReLU(),
            nn.Dropout3d(0.3),

            nn.Linear(in_features=512, out_features=1),
            nn.Sigmoid()

        )

    def forward(self,x):
        x = self.features(x)
        x = self.pool(x)
        x = self.linear(x)
        return x
