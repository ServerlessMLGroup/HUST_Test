import datetime
import logging
import os


def CtxInfo(func_file_name, info):
    current_time = datetime.datetime.now()
    log_file_path = os.getcwd()
    log_file_path += '/logs'
    # 创建日志器对象
    logger = logging.getLogger()

    # 设置logger可输出日志级别范围
    logger.setLevel(logging.DEBUG)

    # 添加控制台handler，用于输出日志到控制台
    console_handler = logging.StreamHandler()
    # 添加日志文件handler，用于输出日志到文件中
    log_file_name = ('%s_%s_%s_%s.txt' %
                     (func_file_name, current_time.month, current_time.day, current_time.hour))
    if not os.path.exists(log_file_path):
        os.mkdir(log_file_path)
    os.path.join(log_file_path, log_file_name)
    f = open(log_file_path + "/" + log_file_name)
    f.close()
    file_handler = logging.FileHandler(filename=log_file_name, encoding='UTF-8')

    # 将handler添加到日志器中
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # 设置格式并赋予handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 输出不同级别日志
    logger.debug("============【开始测试】====================")
    logger.info("============【开始测试】====================")
    logger.warning("============【开始测试】====================")
    logger.critical("============【开始测试】====================")
    logger.error("============【开始测试】====================")

    logger.info(info)