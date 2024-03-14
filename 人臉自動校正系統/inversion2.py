from inversion import *

if __name__ == '__main__':
    # set the memory
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.55
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)
    test_inversion = test_inversion(epochs=1000, warpN=2, num=104, reference_num=117, filename='p_inversion_diff_reference/warp1_model/Limg+Lstyle+Latt/warp2_times')
    test_inversion.p_inversion()