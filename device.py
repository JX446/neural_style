# device_utils.py
import torch

def setup_gpu(params):
    """
    Set the runtime device according to params.gpu and params.backend.
    Returns:
        dtype: default tensor type (CPU/GPU)
        multidevice: whether multiple devices are used
        backward_device: device for backpropagation
    """

    def setup_cuda():
        if 'cudnn' in params.backend:
            torch.backends.cudnn.enabled = True
            if params.cudnn_autotune:
                torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.enabled = False

    def setup_cpu():
        if 'mkl' in params.backend and 'mkldnn' not in params.backend:
            torch.backends.mkl.enabled = True
        elif 'mkldnn' in params.backend:
            raise ValueError("MKL-DNN is not supported yet.")
        elif 'openmp' in params.backend:
            torch.backends.openmp.enabled = True

    multidevice = False
    if "," in str(params.gpu):
        # Multiple devices mode
        devices = params.gpu.split(',')
        multidevice = True

        if 'c' in str(devices[0]).lower():
            backward_device = "cpu"
            setup_cuda(), setup_cpu()
        else:
            backward_device = "cuda:" + devices[0]
            setup_cuda()
        dtype = torch.FloatTensor

    elif "c" not in str(params.gpu).lower():
        # Single device mode
        setup_cuda()
        dtype, backward_device = torch.cuda.FloatTensor, "cuda:" + str(params.gpu)
    else:
        # CPU mode
        setup_cpu()
        dtype, backward_device = torch.FloatTensor, "cpu"

    return dtype, multidevice, backward_device


def setup_multi_device(net, params, ModelParallel):
    assert len(params.gpu.split(',')) - 1 == len(params.multidevice_strategy.split(',')), \
        "The number of -multidevice_strategy layer indices minus 1, must be equal to the number of -gpu devices."
    new_net = ModelParallel(net, params.gpu, params.multidevice_strategy)
    return new_net
