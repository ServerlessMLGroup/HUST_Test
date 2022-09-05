import datetime
import logging
import os

import colorlog as colorlog
console_formatter = colorlog.ColoredFormatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_formatter = colorlog.ColoredFormatter(fmt='[%(asctime)s.%(msecs)03d] %(filename)s -> line:%(lineno)d [%(levelname)s] : %(message)s')


def get_logger():
    # 创建日志器对象
    logger = logging.getLogger()

    # 设置logger可输出日志级别范围
    logger.setLevel(logging.DEBUG)

    # 添加控制台handler，用于输出日志到控制台
    console_handler = logging.StreamHandler()

    # 将handler添加到日志器中
    logger.addHandler(console_handler)

    # 设置格式并赋予handler
    console_handler.setFormatter(console_formatter)
    return logger


def init_file(func_file_name):
    current_time = datetime.datetime.now()
    dir_name, _ = os.path.split(os.path.abspath(__file__))
    log_file_path = os.path.dirname(dir_name) + '/logs'
    log_file_name = ('%s_%s_%s_%s.txt' %
                     (func_file_name, current_time.month, current_time.day, current_time.hour))
    if not os.path.exists(log_file_path):
        os.mkdir(log_file_path)

    file_full_name = log_file_path + "/" + log_file_name
    f = open(file_full_name, "a+")
    f.close()
    return file_full_name


class Log:

    logger = get_logger()

    @staticmethod
    def debug(func_file_name, msg):
        file_full_name = init_file(func_file_name)
        file_handler = logging.FileHandler(filename=file_full_name, encoding='UTF-8')
        Log.logger.addHandler(file_handler)
        file_handler.setFormatter(file_formatter)
        Log.logger.debug(("\033[1;32m {} \033[0m").format(msg))
        Log.logger.removeHandler(file_handler)

    @staticmethod
    def info(func_file_name, msg):
        file_full_name = init_file(func_file_name)
        file_handler = logging.FileHandler(filename=file_full_name, encoding='UTF-8')
        Log.logger.addHandler(file_handler)
        file_handler.setFormatter(file_formatter)
        Log.logger.info(("\033[1;29m {} \033[0m").format(msg))
        Log.logger.removeHandler(file_handler)

    @staticmethod
    def warning(func_file_name, msg):
        file_full_name = init_file(func_file_name)
        file_handler = logging.FileHandler(filename=file_full_name, encoding='UTF-8')
        Log.logger.addHandler(file_handler)
        file_handler.setFormatter(file_formatter)
        Log.logger.warning(("\033[1;33m {} \033[0m").format(msg))
        Log.logger.removeHandler(file_handler)

    @staticmethod
    def error(func_file_name, msg):
        file_full_name = init_file(func_file_name)
        file_handler = logging.FileHandler(filename=file_full_name, encoding='UTF-8')
        Log.logger.addHandler(file_handler)
        file_handler.setFormatter(file_formatter)
        Log.logger.error(("\033[1;31m {} \033[0m").format(msg))
        Log.logger.removeHandler(file_handler)

    @staticmethod
    def critical(func_file_name, msg):
        file_full_name = init_file(func_file_name)
        file_handler = logging.FileHandler(filename=file_full_name, encoding='UTF-8')
        Log.logger.addHandler(file_handler)
        file_handler.setFormatter(file_formatter)
        Log.logger.critical(("\033[1;36m {} \033[0m").format(msg))
        Log.logger.removeHandler(file_handler)
