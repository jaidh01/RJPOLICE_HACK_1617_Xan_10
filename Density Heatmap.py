# Density Heatmap
import torch

def get_density_heatmap(model, image):
    model.eval()
    image = image.unsqueeze(0).float().cuda()
    density_map = model(image)
    density_map = density_map.squeeze().detach().cpu().numpy()
    return density_map

# Count number of people in a sample image
def count_people(density_map):
    return np.sum(density_map)