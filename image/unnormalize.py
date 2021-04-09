class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images):

        images = images.mul_(self.std).add_(self.mean)
            # The normalize code -> t.sub_(m).div_(s)
        return images
      
      
mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
inv_normalize = UnNormalize(mean, std)

row = 4
col = 4
fig, axes = plt.subplots(row, col, figsize=(16, 16))
axes = axes.ravel()

images = next(iter(testloader))
images = inv_normalize(images)
images = images.detach().numpy().transpose(0,2,3,1)

try:
    for j in range(CFG.BATCH):
        axes[j].imshow(images[j])
except:
    print("~Error avoidance~")
