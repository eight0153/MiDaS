import os

# %%
import cv2
import matplotlib.pyplot as plt
import matplotlib.image
import plac
import torch
from torchvision.transforms import Compose

import loaddata_demo as loaddata
import utils

from models.midas_net import MidasNet
from models.transforms import NormalizeImage, PrepareForNet, Resize

plt.set_cmap("gray")


@plac.annotations(
    image_path=plac.Annotation('The path to an RGB image or a directory containing RGB images.', type=str,
                               kind='option', abbrev='i'),
    model_path=plac.Annotation('The path to the pre-trained model weights.', type=str, kind='option', abbrev='m'),
    output_path=plac.Annotation('The path to save the model output to.', type=str, kind='option', abbrev='o'),
)
def main(image_path, model_path='model.pt', output_path=None):
    print("Loading model...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = MidasNet(model_path, non_negative=True)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print("Creating depth maps...")
    rgb_path = os.path.abspath(image_path)

    if os.path.isdir(rgb_path):
        for file in os.listdir(rgb_path):
            test(model, os.path.join(rgb_path, file), output_path)
    else:
        test(model, rgb_path, output_path)

    print("Done.")


def test(model, rgb_path, output_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = Compose(
        [
            Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    img = utils.read_image(rgb_path)
    img_input = transform({"image": img})["image"]

    path, file = os.path.split(rgb_path)
    file = f"{file.split('.')[0]}.png"
    depth_path = os.path.join(output_path, file) if output_path else os.path.join(path, f"out_{file}")

    print(f"{rgb_path} -> {depth_path}")

    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        prediction = model.forward(sample).cpu()

        # Output values are not in valid pixel range (uint8, 0-255) so we need to rescale
        pred_min = prediction.min()
        pred_max = prediction.max()
        prediction = 255 * (prediction - pred_min) / (pred_max - pred_min)

        # Output is inverse depth map, need to compare to normal depth maps so need to flip depth values.
        prediction = (255 - prediction).abs()

        matplotlib.image.imsave(depth_path, prediction.view(prediction.size(1), prediction.size(2)).data.numpy())


if __name__ == '__main__':
    plac.call(main)
