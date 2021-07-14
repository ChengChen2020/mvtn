import torch
import torch.nn as nn
from torchvision.models import resnet

from lib.utils import *

class mvtn(nn.Module):
    def __init__(self, b=40, M=12, num_views=12, distance=3., device=torch.device('cuda:0')):
        super(mvtn, self).__init__()
        self.b = b
        self.M = M
        self.num_views = num_views
        self.distance = torch.tensor(distance)
        self.device = device
        
        self.ubound = (180., 90.)
        
        self.backbone = nn.Sequential(*list(resnet.resnet18(pretrained=True).children())[:-1])
        self.fc = nn.Linear(512, 40)

        # (bs, 2048, 3, 1)
        self.point_net = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=(3,1)),
                nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=1),
                nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=1),
                nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=1),
                nn.BatchNorm2d(128),
            nn.Conv2d(128, 1024, kernel_size=1),
                nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=1),

            nn.Flatten(),

            nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
            nn.Linear(512, 256),
                nn.BatchNorm1d(256), nn.Dropout(p=0.3),
            nn.Linear(256, b),
        )

        self.mvtn_regressor = nn.Sequential(
            nn.Linear(b + 2 * M, b),
                nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(b, b),
                nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(b, 5 * M),
                nn.BatchNorm1d(5 * M), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(5 * M, 2 * M),
                nn.BatchNorm1d(2 * M), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(2 * M, 2 * M),
            nn.Tanh()
        )

    def forward(self, x):
        # MVTN
        # (bs, 2048, 3)
        # x = x.to(device)
        bs = x.size()[0]

        point_clouds = [Pointclouds(
            points=[x[i]],
            features=[torch.ones(2048, 3).to(self.device)],
        ) for i in range(bs)]

        x = x.unsqueeze(-1)
        x = self.point_net(x)
        # Random initialization for scene parameters
        b = torch.zeros(bs, 2 * self.M).to(self.device)
        x = torch.cat((x, b), dim=1)
        
        # Renderer
        scene_parameters = self.mvtn_regressor(x)
        elevation = scene_parameters[:, 0::2] * self.ubound[0]
        azimuth = scene_parameters[:, 1::2] * self.ubound[1]
        
        # https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/renderer/cameras.html#look_at_view_transform
        RT = [look_at_view_transform(
            self.distance, 
            elevation[i], 
            azimuth[i], 
            device=self.device
        ) for i in range(bs)]
        # print(R.shape, T.shape)
        # image = phong_renderer(meshes_world=meshes.clone().extend(self.num_views), R=R, T=T)

        images = [points_renderer(self.device)(
            point_clouds[i].clone().extend(self.num_views), 
            R=RT[i][0],
            T=RT[i][1],
        ) for i in range(bs)]
        # demo = images[0].detach().cpu().numpy()

        images = torch.cat(images)
        # print(images.shape)

        # MVC
        mv = images[..., :3].permute(0, 3, 1, 2)
        y = self.backbone(mv)
        y = y.view((int(mv.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))
        y = self.fc(torch.max(y,1)[0].view(y.shape[0],-1))

        return y
