import torch
from PIL import Image
from torchvision import transforms, models
import torch.optim as optim
import torchvision.utils as vutils

# Input
content_img = Image.open("input.jpg")
style_img = Image.open("style.jpg")

# Hyper-parameters
NUM_UPDATES = 1000  # Might need more for random start
LR = 0.01
ALPHA = 1e4
BETA = 1e9
LAYERS = [0, 5, 10, 19, 28]
AVG_POOL = True
START_RANDOM = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Images
c_h, c_w = content_img.size
s_h, s_w = style_img.size
min_h = min(c_h, s_h)
min_w = min(c_w, s_w)

get_tensor = transforms.Compose([
    transforms.CenterCrop([min_w, min_h]),
    transforms.ToTensor()])

content_img = get_tensor(content_img).unsqueeze(0).to(device)
style_img = get_tensor(style_img).unsqueeze(0).to(device)

if START_RANDOM:
    generated_img = torch.rand(content_img.shape, requires_grad=True, device=device)
else:
    generated_img = content_img.clone().requires_grad_(True).to(device)

# Optimizer
optimizer = optim.Adam([generated_img], lr=LR)

# Model
vgg = models.vgg19(pretrained=True).features.to(device).eval()

if AVG_POOL:
    for i, layer in enumerate(vgg):
        if isinstance(layer, torch.nn.MaxPool2d):
            vgg[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2)

for param in vgg.parameters():
    param.requires_grad_(False)

# ImageNet Mean & Std
mean = torch.Tensor([0.485, 0.456, 0.406]).reshape([1, -1, 1, 1]).to(device)
std = torch.Tensor([0.229, 0.224, 0.225]).reshape([1, -1, 1, 1]).to(device)


def get_vgg_features(img):
    features = []
    x = img
    x = (x - mean) / std
    for layer_idx, m in enumerate(vgg):
        x = m(x)
        if layer_idx in LAYERS:
            features.append(x.reshape([x.shape[1], -1]))
    return features


content_features = get_vgg_features(content_img)
style_features = get_vgg_features(style_img)

for i in range(NUM_UPDATES + 1):
    content_loss = 0
    style_loss = 0

    generated_features = get_vgg_features(generated_img)

    for j in range(len(LAYERS)):
        if j == (len(LAYERS) - 1):
            content_loss += ((content_features[j] - generated_features[j]) ** 2).mean() * ALPHA

        content_gram = torch.mm(style_features[j], style_features[j].t())
        generated_gram = torch.mm(generated_features[j], generated_features[j].t())
        style_loss_ = ((content_gram - generated_gram) ** 2).mean()
        style_loss_ = style_loss_ / (4 * (style_features[j].shape[0] * style_features[j].shape[1]) ** 2)
        style_loss += style_loss_ * BETA

    total_loss = style_loss + content_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if i % 50 == 0:
        print("Step: " + str(i)
              + "\tContent_loss:" + str(content_loss.item())
              + "\tStyle loss:" + str(style_loss.item()))
        vutils.save_image(generated_img.clone(), 'sample_' + str(i) + '.png')
