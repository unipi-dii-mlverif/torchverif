import torch
import torch.nn as nn

num = -132076555962054721296208061/1000000000000000000000000000
model = torch.load("test_model.pth", map_location=torch.device('cpu'))

print(model)
print("num: ", num)
print("sigmoid(num): ", torch.sigmoid(torch.tensor([num])).item())

print(model(torch.tensor([0.0, 0.0, 0.0, 0.0])))
torch.manual_seed(9999)
# Define the model
model = nn.Sequential(
    nn.Linear(4, 10, bias=True),
    nn.ReLU(),
    nn.Linear(10, 2, bias=True)
)

torch.save(model, "test_model.pth")
