# train_utils.py
import torch.optim as optim

def setup_optimizer(img, params):
    """
    根据参数设置优化器
    返回:
        optimizer: torch.optim 优化器对象
        loopVal: 外层循环次数控制变量
    """
    if params.optimizer == 'lbfgs':
        print("Running optimization with L-BFGS")
        optim_state = {
            'max_iter': params.num_iterations,
            'tolerance_change': -1,
            'tolerance_grad': -1,
        }
        if params.lbfgs_num_correction != 100:
            optim_state['history_size'] = params.lbfgs_num_correction
        optimizer = optim.LBFGS([img], **optim_state)
        loopVal = 1

    elif params.optimizer == 'adam':
        print("Running optimization with ADAM")
        optimizer = optim.Adam([img], lr=params.learning_rate)
        loopVal = params.num_iterations - 1

    return optimizer, loopVal
