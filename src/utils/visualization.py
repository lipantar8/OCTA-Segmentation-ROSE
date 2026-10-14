import matplotlib.pyplot as plt
import seaborn as sns
import torch


def show_predictions(model, loader, device, n=4):
    """
    Visualize model predictions compared with input images and ground truth masks.

    Args:
        model: Trained segmentation model.
        loader: DataLoader containing images and masks.
        device: Computation device.
        n: Number of samples to display.
    """

    model.eval()

    images, masks = next(iter(loader))

    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)
        preds = torch.sigmoid(outputs)


    images = images.cpu()
    masks = masks.cpu()
    preds = preds.cpu()


    n = min(n, images.size(0))

    plt.figure(figsize=(12, 4 * n))


    for i in range(n):

        img = images[i].permute(1, 2, 0).numpy()
        mask = masks[i].squeeze().numpy()

        pred_prob = preds[i].squeeze().numpy()
        pred_mask = (pred_prob > 0.5).astype(float)


        plt.subplot(n, 3, i * 3 + 1)
        plt.imshow(img)
        plt.title("Image")
        plt.axis("off")


        plt.subplot(n, 3, i * 3 + 2)
        plt.imshow(
            mask,
            cmap="gray",
            vmin=0,
            vmax=1
        )
        plt.title("Ground Truth")
        plt.axis("off")


        plt.subplot(n, 3, i * 3 + 3)
        plt.imshow(
            pred_mask,
            cmap="gray",
            vmin=0,
            vmax=1
        )
        plt.title("Prediction")
        plt.axis("off")


    plt.tight_layout()
    plt.show()