import segmentation_models_pytorch as smp

def get_attention_unet():
    """
    Create an Attention U-Net model with a ResNet34 encoder.

    Returns:
        torch.nn.Module: Attention U-Net model.
    """
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
        decoder_attention_type="scse"
    )

    return model