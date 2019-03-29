# MLT
多任务深度学习网络。
MLT_Lee.py适用于windows10下的python3.6 torch 0.4,若在linux下使用，可修改dataloader函数中的numworker，多进程生成数据集。
在初始版本中，BATCH_SIZE_SOURCE = 10,BATCH_SIZE_TARGET = 2,即使这么小，经试验1050TI扔不能运行。cuda:out of memory.可在更大显存的显卡中运行。
在实际中，尽量不要让BATCH_SIZE这么小。
