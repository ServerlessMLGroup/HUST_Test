import sys

sys.path.append("..")

from util import log


if __name__ == '__main__':
    logger = log.get_logger("log-test")
    logger.debug("this is debug")
    logger.info("this is info")
    logger.warning("this is warning")
    logger.error("this is error")
    logger.critical("this is critical")
