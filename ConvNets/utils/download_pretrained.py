import torch
import torchvision.models as models

model = models.densenet201(pretrained=True)
torch.save(model.state_dict(), 'densenet201.pth')
