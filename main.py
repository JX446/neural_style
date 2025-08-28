import torch
import torch.nn as nn

from config import get_params
from device import setup_gpu
from image import preprocess, prepare_style_inputs, save_output
from losses import normalize_weights
from model import loadmodel
from style_transfer_network import build_style_transfer_network
from optimizer import setup_optimizer

params = get_params()

def main():
    dtype, multidevice, backward_device = setup_gpu(params)
    cnn, layerList = loadmodel(params.model_file, params.pooling, params.gpu, params.disable_check)
    content_image_preprocessed = preprocess(params.content_image, params.image_size).type(dtype)
    style_images_preprocessed, init_image, blend_weights = prepare_style_inputs(params, content_image_preprocessed, dtype=dtype)

    net, content_losses, style_losses, tv_losses = build_style_transfer_network(cnn, layerList,params, dtype)

    # Capture content targets
    for i in content_losses:
        i.mode = 'capture'
    print("Capturing content targets")
    net(content_image_preprocessed)

    # Capture style targets
    for i in content_losses:
        i.mode = 'None'

    for i, image in enumerate(style_images_preprocessed):
        print("Capturing style target " + str(i+1))
        for j in style_losses:
            j.mode = 'capture'
            j.blend_weight = blend_weights[i]
        net(style_images_preprocessed[i])

    # Set all loss modules to loss mode
    for i in content_losses:
        i.mode = 'loss'
    for i in style_losses:
        i.mode = 'loss'

    # Maybe normalize content and style weights
    if params.normalize_weights:
        normalize_weights(content_losses, style_losses)

    # Freeze the network in order to prevent
    # unnecessary gradient calculations
    for param in net.parameters():
        param.requires_grad = False

    # Initialize the image
    if params.seed >= 0:
        torch.manual_seed(params.seed)
        torch.cuda.manual_seed_all(params.seed)
        torch.backends.cudnn.deterministic=True
    if params.init == 'random':
        B, C, H, W = content_image_preprocessed.size()
        img = torch.randn(C, H, W).mul(0.001).unsqueeze(0).type(dtype)
    elif params.init == 'image':
        if params.init_image != None:
            img = init_image.clone()
        else:
            img = content_image_preprocessed.clone()
    img = nn.Parameter(img)

    # Function to evaluate loss and gradient. We run the net forward and
    # backward to get the gradient, and sum up losses from the loss modules.
    # optim.lbfgs internally handles iteration and calls this function many
    # times, so we manually count the number of iterations to handle printing
    # and saving intermediate results.
    num_calls = [0]
    def feval():
        num_calls[0] += 1
        optimizer.zero_grad()
        net(img)
        loss = 0

        for mod in content_losses:
            loss += mod.loss.to(backward_device)
        for mod in style_losses:
            loss += mod.loss.to(backward_device)
        if params.tv_weight > 0:
            for mod in tv_losses:
                loss += mod.loss.to(backward_device)

        loss.backward()
        save_output(params, num_calls[0], img, content_image_preprocessed)
        return loss
    
    # train
    optimizer, loopVal = setup_optimizer(img, params)
    while num_calls[0] <= loopVal:
         optimizer.step(feval)

if __name__ == "__main__":
    main()    