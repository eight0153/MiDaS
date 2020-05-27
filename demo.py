import os

# %%
import cv2
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import plac
import torch
from torchvision.transforms import Compose

# import loaddata_demo as loaddata
import MiDaS.utils as utils

from .models.midas_net import MidasNet
from .models.transforms import NormalizeImage, PrepareForNet, Resize

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
        min_depths = []
        max_depths = []

        for file in os.listdir(rgb_path):
            depth_map = test(model, os.path.join(rgb_path, file), output_path)

            min_depths.append(depth_map.min())
            max_depths.append(depth_map.max())

        print("Min Depth: {.2f}".format(min(min_depths)))
        print("Max Depth: {.2f}".format(max(max_depths)))
        print("Avg. Min Depth: {.2f}".format(np.mean(min_depths)))
        print("Avg. Max Depth: {.2f}".format(np.mean(max_depths)))
    else:
        test(model, rgb_path, output_path)

    print("Done.")


def test(model, rgb_path, output_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = Compose(
        [
            Resize(
                1920,
                1080,
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
    file = "{}.png".format(file.split('.')[0])
    depth_path = os.path.abspath(os.path.join(output_path, file) if output_path else os.path.join(path, "out_{}".format(file)))

    os.makedirs(os.path.dirname(depth_path), exist_ok=True)

    print("{} -> {}".format(rgb_path, depth_path))

    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        prediction = model.forward(sample)

        output = prediction.permute((1, 2, 0))
        # output = 1/1000 * (10000 - output)
        output = output.squeeze().cpu().numpy()

        # np.save(depth_path, output)
        # Convert to the range [0, 1] so that matplotlib doesn't automatically scale the depth values.
        # matplotlib.image.imsave(depth_path, 1/10000 * output)
        cv2.imwrite(depth_path, 1/10000 * output)

    return output


if __name__ == '__main__':
    plac.call(main)
