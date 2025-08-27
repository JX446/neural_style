# device_utils.py
import torch

def setup_gpu(params):
    """
    根据 params.gpu 和 params.backend 设置运行设备
    返回:
        dtype: 默认张量类型 (CPU/GPU)
        multidevice: 是否为多设备
        backward_device: 反向传播所在设备
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
        # 多设备模式
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
        # 单 GPU 模式
        setup_cuda()
        dtype, backward_device = torch.cuda.FloatTensor, "cuda:" + str(params.gpu)
    else:
        # CPU 模式
        setup_cpu()
        dtype, backward_device = torch.FloatTensor, "cpu"

    return dtype, multidevice, backward_device


def setup_multi_device(net, params, ModelParallel):
    """
    构建多设备网络
    """
    assert len(params.gpu.split(',')) - 1 == len(params.multidevice_strategy.split(',')), \
        "The number of -multidevice_strategy layer indices minus 1, must be equal to the number of -gpu devices."
    new_net = ModelParallel(net, params.gpu, params.multidevice_strategy)
    return new_net
