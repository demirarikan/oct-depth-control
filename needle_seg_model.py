import monai
import numpy as np
import torch
from torchvision import transforms
import cv2


class NeedleSegModel():
    def __init__(self, weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=4,
            channels=(16, 32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2, 2),
            kernel_size=3,
        ).to(self.device)

        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()
        print('Model loaded successfully')

    def prepare_vol_from_leica_engine(self, oct_volume):
        oct_volume = oct_volume.transpose(1, 0, 2)
        oct_volume = np.rot90(oct_volume, axes=(1, 2))
        oct_volume = oct_volume.astype(np.float32)
        oct_volume = torch.tensor(oct_volume).to(self.device)
        oct_volume = transforms.Pad((12, 0, 12, 0))(oct_volume)
        oct_volume = oct_volume.unsqueeze(1)
        return oct_volume
    
    def segment_volume(self, oct_volume, debug=False):
        with torch.no_grad():
            seg_volume = self.model(oct_volume)
            seg_volume = torch.argmax(torch.softmax(seg_volume, dim=1), dim=1)
            seg_volume = seg_volume.cpu().numpy()
            if debug:
                for idx, seg_mask in enumerate(seg_volume):
                    seg_mask = apply_color_map(seg_mask)
                    cv2.imwrite(f'bscan_{idx}.png', seg_mask)
        return seg_volume
    
def apply_color_map(seg_mask):
    # Create an empty image with 3 channels (RGB)
    color_image = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)
    
    # Define colors for each segment value (0: black, 1: red, 2: green, 3: blue)
    colors = {
        0: [0, 0, 0],        # black for background
        1: [255, 0, 0],      # red
        2: [0, 255, 0],      # green
        3: [0, 0, 255],      # blue
        # Add more colors if there are more segments
    }
    
    # Apply the colors to the image
    for val, color in colors.items():
        color_image[seg_mask == val] = color

    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    return color_image