from AMTS import AMTS
from utils import logger
from args import config

def log_info(logger, config):

    logger.info('--- Configs List---')
    logger.info('--- Dadaset:{}'.format(config.DATASET))
    logger.info('--- Train:{}'.format(config.TRAIN))
    logger.info('--- Bit:{}'.format(config.HASH_BIT))
    logger.info('--- Eta:{}'.format(config.eta))
    logger.info('--- Beta:{}'.format(config.beta))
    logger.info('--- Lambda:{}'.format(config.lamb))
    logger.info('--- Mu:{}'.format(config.mu))
    logger.info('--- delta:{}'.format(config.delta))
    logger.info('--- epsilon:{}'.format(config.epsilon))
    logger.info('--- phi:{}'.format(config.phi))
    logger.info('--- gamma:{}'.format(config.gamma))
    logger.info('--- Batch:{}'.format(config.BATCH_SIZE))
    logger.info('--- Lr_IMG:{}'.format(config.LR_IMG))
    logger.info('--- Lr_TXT:{}'.format(config.LR_TXT))


def main():

        # log
        log = logger()
        log_info(log, config)
        # if config.TRAIN == False:
        #     Model.load_checkpoints(config.CHECKPOINT)
        #     Model.eval()
        #
        # else:
        for lamb1 in [0.001, 0.01, 0.1, 1, 3, 5, 7, 10]:
            for lamb2 in [0.001, 0.01, 0.1, 1, 3, 5, 7, 10]:
                config.lamb1 = lamb1
                config.lamb2 = lamb2
                log_info(log, config)
                Model = RME(log, config)
                for epoch in range(config.NUM_EPOCH):
                    Model.train(epoch)
                    if (epoch + 1) % config.EVAL_INTERVAL == 0:
                        Model.eval()
                        log.handlers.clear()
                    # save the model
                    # if epoch + 1 == config.NUM_EPOCH:
                    #     Model.save_checkpoints(config.CHECKPOINT)



if __name__ == '__main__':
    main()
