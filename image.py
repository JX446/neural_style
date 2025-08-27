# image_utils.py
import os
import torch
import torchvision.transforms as transforms
from PIL import Image


# Preprocess an image: load → resize → convert to tensor → RGB to BGR → subtract mean
def preprocess(image_name, image_size):
    image = Image.open(image_name).convert('RGB')

    # If image_size is not a tuple, scale proportionally
    if type(image_size) is not tuple:
        image_size = tuple([
            int((float(image_size) / max(image.size)) * x) 
            for x in (image.height, image.width)
        ])

    # Resize + ToTensor
    Loader = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    # RGB → BGR
    rgb2bgr = transforms.Compose([
        transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])
    ])

    # Subtract mean (same as VGG training)
    Normalize = transforms.Compose([
        transforms.Normalize(mean=[103.939, 116.779, 123.68], std=[1,1,1])
    ])

    tensor = Normalize(rgb2bgr(Loader(image) * 255)).unsqueeze(0)
    return tensor


# Deprocess: convert tensor back to displayable PIL image
def deprocess(output_tensor):
    Normalize = transforms.Compose([
        transforms.Normalize(mean=[-103.939, -116.779, -123.68], std=[1,1,1])
    ])
    bgr2rgb = transforms.Compose([
        transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])
    ])

    output_tensor = bgr2rgb(Normalize(output_tensor.squeeze(0).cpu())) / 255
    output_tensor.clamp_(0, 1)

    Image2PIL = transforms.ToPILImage()
    image = Image2PIL(output_tensor.cpu())
    return image


# Postprocess to preserve original colors in style transfer
def original_colors(content, generated):
    content_channels = list(content.convert('YCbCr').split())
    generated_channels = list(generated.convert('YCbCr').split())
    content_channels[0] = generated_channels[0]  # replace Y channel
    return Image.merge('YCbCr', content_channels).convert('RGB')


# Load multiple style images (supports directories and multiple files)
def load_style_images(style_image_input, image_size, style_scale=1.0):
    style_image_list, ext = [], [".jpg", ".jpeg", ".png", ".tiff"]

    for image in style_image_input.split(','):
        if os.path.isdir(image):
            images = (
                os.path.join(image, file) for file in os.listdir(image)
                if os.path.splitext(file)[1].lower() in ext
            )
            style_image_list.extend(images)
        else:
            style_image_list.append(image)

    style_images = []
    for image in style_image_list:
        style_size = int(image_size * style_scale)
        img = preprocess(image, style_size)
        style_images.append(img)

    return style_images, style_image_list


def prepare_style_inputs(params, content_image=None, dtype=torch.FloatTensor):
    """
    Prepare style images, initial image (optional), and style blending weights for style transfer.

    Args:
        params: Namespace with user parameters (style_image, init_image, style_scale, image_size, style_blend_weights).
        content_image: Content image Tensor (needed if init_image is provided).
        dtype: Torch Tensor type (FloatTensor or CUDA tensor).

    Returns:
        style_images_preprocessed: List of preprocessed style image tensors.
        init_image: Preprocessed initial image tensor (or None).
        style_blend_weights: Normalized list of blending weights.
    """
    # 1. Collect style image paths
    style_image_input = params.style_image.split(',')
    style_images_original = []
    ext = [".jpg", ".jpeg", ".png", ".tiff"]
    
    for image in style_image_input:
        if os.path.isdir(image):
            images = (os.path.join(image, file) for file in os.listdir(image)
                      if os.path.splitext(file)[1].lower() in ext)
            style_images_original.extend(images)
        else:
            style_images_original.append(image)
    
    # 2. Preprocess style images
    style_images_preprocessed = []
    for image in style_images_original:
        style_size = int(params.image_size * params.style_scale)
        img = preprocess(image, style_size).type(dtype)
        style_images_preprocessed.append(img)
    
    # 3. Preprocess initial image (if any)
    init_image = None
    if params.init_image is not None:
        if content_image is None:
            raise ValueError("content_image must be provided if init_image is used.")
        image_size = (content_image.size(2), content_image.size(3))
        init_image = preprocess(params.init_image, image_size).type(dtype)
    
    # 4. Handle style blending weights
    if params.style_blend_weights is None:
        style_blend_weights = [1.0] * len(style_images_original)
    else:
        style_blend_weights = [float(w) for w in params.style_blend_weights.split(',')]
        assert len(style_blend_weights) == len(style_images_original), \
            "-style_blend_weights and -style_images must have the same number of elements!"
    
    # 5. Normalize blending weights to sum to 1
    weight_sum = sum(style_blend_weights)
    style_blend_weights = [w / weight_sum for w in style_blend_weights]
    
    return style_images_preprocessed, init_image, style_blend_weights
