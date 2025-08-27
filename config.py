import argparse

def get_params():
    parser = argparse.ArgumentParser()
    
    # Basic options
    parser.add_argument("-style_image", help="Style target image", default='examples/inputs/seated-nude.jpg')
    parser.add_argument("-style_blend_weights", default=None)
    parser.add_argument("-content_image", help="Content target image", default='examples/inputs/tubingen.jpg')
    parser.add_argument("-image_size", help="Maximum height / width of generated image", type=int, default=512)
    parser.add_argument("-gpu", help="Zero-indexed ID of the GPU to use; for CPU mode set -gpu = c", default=0)

    # Optimization options
    parser.add_argument("-content_weight", type=float, default=5e0)
    parser.add_argument("-style_weight", type=float, default=1e2)
    parser.add_argument("-normalize_weights", action='store_true')
    parser.add_argument("-normalize_gradients", action='store_true')
    parser.add_argument("-tv_weight", type=float, default=1e-3)
    parser.add_argument("-num_iterations", type=int, default=1000)
    parser.add_argument("-init", choices=['random', 'image'], default='random')
    parser.add_argument("-init_image", default=None)
    parser.add_argument("-optimizer", choices=['lbfgs', 'adam'], default='lbfgs')
    parser.add_argument("-learning_rate", type=float, default=1e0)
    parser.add_argument("-lbfgs_num_correction", type=int, default=100)

    # Output options
    parser.add_argument("-print_iter", type=int, default=50)
    parser.add_argument("-save_iter", type=int, default=100)
    parser.add_argument("-output_image", default='out.png')

    # Other options
    parser.add_argument("-style_scale", type=float, default=1.0)
    parser.add_argument("-original_colors", type=int, choices=[0, 1], default=0)
    parser.add_argument("-pooling", choices=['avg', 'max'], default='max')
    parser.add_argument("-model_file", type=str, default='models/vgg19-d01eb7cb.pth')
    parser.add_argument("-disable_check", action='store_true')
    parser.add_argument("-backend", choices=['nn', 'cudnn', 'mkl', 'mkldnn', 'openmp', 'mkl,cudnn', 'cudnn,mkl'], default='nn')
    parser.add_argument("-cudnn_autotune", action='store_true')
    parser.add_argument("-seed", type=int, default=-1)

    parser.add_argument("-content_layers", help="layers for content", default='relu4_2')
    parser.add_argument("-style_layers", help="layers for style", default='relu1_1,relu2_1,relu3_1,relu4_1,relu5_1')

    parser.add_argument("-multidevice_strategy", default='4,7,29')
    
    params = parser.parse_args()
    return params