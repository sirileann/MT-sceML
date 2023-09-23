import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.module import Embedding, NeighborEmbedding, OA, SA, CrossAttentionLayer


class NaivePCT(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = Embedding(3, 128)

        self.sa1 = SA(128)
        self.sa2 = SA(128)
        self.sa3 = SA(128)
        self.sa4 = SA(128)

        self.linear = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x = self.embedding(x)

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean


class SPCT(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = Embedding(3, 128)

        self.sa1 = OA(128)
        self.sa2 = OA(128)
        self.sa3 = OA(128)
        self.sa4 = OA(128)

        self.linear = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x = self.embedding(x)

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean


class PCT(nn.Module):
    def __init__(self, samples=[512, 256]):
        super().__init__()

        self.neighbor_embedding = NeighborEmbedding(samples)

        self.oa1 = OA(256)
        self.oa2 = OA(256)
        self.oa3 = OA(256)
        self.oa4 = OA(256)

        self.linear = nn.Sequential(
            nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x = self.neighbor_embedding(x)

        x1 = self.oa1(x)
        x2 = self.oa2(x1)
        x3 = self.oa3(x2)
        x4 = self.oa4(x3)

        x = torch.cat([x, x1, x2, x3, x4], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean


class Classification(nn.Module):
    def __init__(self, num_categories=30):
        super().__init__()

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, num_categories)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class Classification_twoInputs(nn.Module):
    def __init__(self, num_categories=30):
        super().__init__()

        self.linear0 = nn.Linear(2048, 1024, bias=False)  # added since there were concatenated features
        self.linear1 = nn.Linear(1024, 512)  # removed bias=False
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, num_categories)

        self.bn0 = nn.BatchNorm1d(1024)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.dp0 = nn.Dropout(p=0.5)
        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.bn0(self.linear0(x)))
        x = self.dp0(x)
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class Classification_threeInputs(nn.Module):
    def __init__(self, num_categories=30):
        super().__init__()

        self.linear_init = nn.Linear(3072, 2048, bias=False)  # added since there were concatenated features
        self.linear0 = nn.Linear(2048, 1024)  # added since there were concatenated features
        self.linear1 = nn.Linear(1024, 512)  # removed bias=False
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, num_categories)

        self.bn_init = nn.BatchNorm1d(2048)
        self.bn0 = nn.BatchNorm1d(1024)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.dp0 = nn.Dropout(p=0.5)
        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.bn_init(self.linear_init(x)))
        x = self.dp0(x)
        x = F.relu(self.bn0(self.linear0(x)))
        x = self.dp0(x)
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x


"""
Single-Modal networks.
"""


class PointCloudTransformer_PCT(nn.Module):
    def __init__(self, num_categories=30):
        super().__init__()

        self.encoder = PCT()
        self.cls = Classification(num_categories)

    def forward(self, x):
        _, x, _ = self.encoder(x)
        x = self.cls(x)
        return x


class PointCloudTransformer_SPCT(nn.Module):
    def __init__(self, num_categories=30):
        super().__init__()

        self.encoder = SPCT()
        self.cls = Classification(num_categories)

    def forward(self, x):
        _, x, _ = self.encoder(x)
        x = self.cls(x)
        return x


class PointCloudTransformer_NaivePCT(nn.Module):
    def __init__(self, num_categories=40):
        super().__init__()

        self.encoder = NaivePCT()
        self.cls = Classification(num_categories)

    def forward(self, x):
        _, x, _ = self.encoder(x)
        x = self.cls(x)
        return x


"""
Multi-Modal Networks.
"""


class MultiPCT_Back_Fix(nn.Module):
    def __init__(self, num_categories=30, use_cross_attention=False):
        super().__init__()

        self.encoder = PCT()
        self.encoder2 = SPCT()

        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attention = CrossAttentionLayer(query_dim=1024, key_dim=1024, value_dim=1024, output_dim=1024)

        self.cls = Classification_twoInputs(num_categories)

    def forward(self, x, y):
        _, x, _ = self.encoder(x)
        _, y, _ = self.encoder2(y)

        if self.use_cross_attention:
            # Apply cross-attention to fuse features from the point cloud and image modalities
            x_fused = self.cross_attention(x, y, y)
            x = torch.cat([x, x_fused], dim=1)

        else:
            # # concatenate without cross-attention
            x = torch.cat([x, y], dim=1)

        x = self.cls(x)
        return x


class MultiPCT_BackPC_BackDM(nn.Module):
    def __init__(self, num_categories=30, use_cross_attention=False):
        super().__init__()

        self.encoder = PCT()
        self.cnn = self.get_model()

        self.use_cross_attention = use_cross_attention
        self.cross_attention = CrossAttentionLayer(query_dim=1024, key_dim=1024, value_dim=1024, output_dim=1024)

        self.cls = Classification_twoInputs(num_categories)

    def get_model(self):
        model = models.DenseNet()

        # Changing from 3-channel input layer to 1-channel input layer
        new_conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        new_conv.weight = nn.Parameter(model.features.conv0.weight[:, 0:1, :, :])
        model.features.conv0 = new_conv

        # Modify the last fully connected layer to have output size [1, 1024]
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 1024)

        return model

    def forward(self, x, y):
        _, x, _ = self.encoder(x)
        y = self.cnn(y)

        if self.use_cross_attention:
            # Apply cross-attention to fuse features from the point cloud and image modalities
            x_fused = self.cross_attention(x, y, y)
            x = torch.cat([x, x_fused], dim=1)

        else:
            # # concatenate without cross-attention
            x = torch.cat([x, y], dim=1)

        x = self.cls(x)
        return x


class MultiPCT_Back_ESL_Fix(nn.Module):
    def __init__(self, num_categories=30):
        super().__init__()

        self.encoder = PCT()
        self.encoder2 = SPCT()

        self.cls = Classification_threeInputs(num_categories)

    def forward(self, x, y, z):
        _, x, _ = self.encoder(x)
        _, y, _ = self.encoder2(y)
        _, z, _ = self.encoder2(z)

        x = torch.cat([x, y, z], dim=1)

        x = self.cls(x)
        return x
