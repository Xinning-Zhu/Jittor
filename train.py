import time
import jittor as jt
from options.train_options import TrainOptions
from data import create_dataset, get_test_loaders
from models import create_model  
from util.visualizer import Visualizer
from util.util import write_images
from models.utils import eval_loader, SimpleLogger
import os


jt.flags.use_cuda = 1

if __name__ == '__main__':
    opt = TrainOptions().parse()   # 获取训练参数
    dataset = create_dataset(opt)  # 创建数据集
    dataset_size = len(dataset)    # 获取数据集大小
    test_loader_a, test_loader_b = get_test_loaders(opt)


    # 准备固定的测试数据
    fix_a_data = [test_loader_a.dataset[i]['A'] for i in range(opt.display_size)]
    fix_a = jt.stack(fix_a_data)
    fix_b_data = [test_loader_b.dataset[i]['A'] for i in range(opt.display_size)]
    fix_b = jt.stack(fix_b_data)

    model = create_model(opt)      # 创建模型
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # 创建可视化工具
    opt.visualizer = visualizer
    # 创建一个日志记录器 SimpleLogger，并指定日志文件路径（opt.run_dir/test.txt）。
    # 用于记录测试 / 评估阶段的关键指标，方便后续分析模型性能变化。
    test_logger = SimpleLogger(os.path.join(opt.run_dir, 'test.txt'))
    total_iters = 0                # 总训练迭代次数，在训练中用于触发特定操作

    optimize_time = 0.1
    #初始化一个变量，用于平滑记录每次迭代的优化时间（即模型前向传播 + 反向传播 + 参数更新的耗时）。

    times = []
    # 外层循环：不同的epochs
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    
        epoch_start_time = time.time()  # 整个epoch的计时器
        iter_data_time = time.time()    # 每次迭代数据加载的计时器
        epoch_iter = 0                  # 当前epoch的训练迭代次数
        visualizer.reset()              # 重置可视化工具

        dataset.set_epoch(epoch)
        # 内层循环：每个epoch中的迭代
        for i, data in enumerate(dataset):  
            iter_start_time = time.time()  # 每次迭代计算的计时器
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].shape[0] 
            total_iters += batch_size
            epoch_iter += batch_size
            optimize_start_time = time.time()
            
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               
            
            model.set_input(data)  
            model.optimize_parameters()  
            
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            # 打印训练损失并保存日志
            if total_iters % opt.print_freq == 0:    
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            # 保存最新模型
            if total_iters % opt.save_latest_freq == 0:   
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # 显示和保存图像
        if epoch % (opt.eval_epoch_freq) == 0 or epoch == 1:  
            images = model.interpolation(fix_a, fix_b)
            write_images(images, 12, opt.img_dir, postfix='%03d_interp' % epoch)
            images = model.sample(fix_a, fix_b)
            write_images(images, opt.display_size, opt.img_dir, postfix='%03d_sample' % epoch)

        # 保存模型
        if epoch % opt.save_epoch_freq == 0:              
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        # 评估模型
        if epoch % opt.eval_epoch_freq == 0:
            results = eval_loader(model, test_loader_a, test_loader_b, opt.run_dir, opt)
            test_logger.log(epoch, opt.n_epochs+opt.n_epochs_decay, results, verbose=True)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()  # 在每个epoch结束时更新学习率
