# import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" #本行只是在用pycharm dubug的时候用到--->注意①:一定一定要把此行代码放在所有访问GPU的代码之前，否则，这里不管设置什么,都是多卡。注意②:在终端运行的时候,需要注释掉该行，在终端选择需要的gpu卡号。
import torch
import json
import torch.backends.cudnn as cudnn
from config import args
from trainer import Trainer
from logger_config import logger

# print('===========================>', torch.cuda.current_device())

def main():
    ngpus_per_node = torch.cuda.device_count()
    cudnn.benchmark = True

    logger.info("Use {} gpus for training".format(ngpus_per_node))

    trainer = Trainer(args, ngpus_per_node=ngpus_per_node)
    logger.info('Args={}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
    trainer.train_loop()


if __name__ == '__main__':
    main()
