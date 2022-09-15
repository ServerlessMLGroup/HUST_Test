import datetime
import logging
import os
import sys

loggers = {}

LOG_ENABLED = True  # 是否开启日志
LOG_TO_CONSOLE = True  # 是否输出到控制台
LOG_TO_FILE = True  # 是否输出到文件
LOG_TO_ES = False  # 是否输出到 Elasticsearch

LOG_LEVEL = 'DEBUG'  # 日志级别
LOG_FORMAT = '[%(levelname)s %(asctime)s] - test:%(name)s - process:%(process)d - file:%(filename)s - line:%(lineno)d ' \
             '- %(message)s'  # 每条日志输出格式
APP_ENVIRONMENT = 'dev'  # 运行环境，如测试环境还是生产环境


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


def get_logger(name=None, func_name=None):
    """
    get logger by name

    :param name: name of logger
    :return: logger
    """
    global loggers

    if not name: name = __name__

    if loggers.get(name):
        return loggers.get(name)

    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # 输出到控制台
    if LOG_ENABLED and LOG_TO_CONSOLE:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=LOG_LEVEL)
        formatter = logging.Formatter(LOG_FORMAT)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # 输出到文件
    if LOG_ENABLED and LOG_TO_FILE:
        file_full_name = init_file(func_name)
        # 添加 FileHandler
        file_handler = logging.FileHandler(file_full_name, encoding='utf-8')
        file_handler.setLevel(level=LOG_LEVEL)
        formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 保存到全局 loggers
    loggers[name] = logger
    return logger
