import logger
from argument_parser import setup_argument_parser
from dataset import create_dataset
import inception_model as inception

log = logger.setup_logger(__name__)

def main():
    config = setup_argument_parser()
    log.info("Starting...")
    log.info("Model will train with following parameters:")
    log.info(config)

    #Create the dataset:
    train_ds, val_ds = create_dataset(config.augmentation)
    model = inception.create_model(config, train_ds, val_ds)


if __name__ == "__main__":
    main()
