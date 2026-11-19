"""RGB normalization shared by training and evaluation."""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalization_parameters(name="imagenet"):
    if name != "imagenet":
        raise ValueError("HG-CrackSeg requires ImageNet RGB normalization")
    return IMAGENET_MEAN, IMAGENET_STD
