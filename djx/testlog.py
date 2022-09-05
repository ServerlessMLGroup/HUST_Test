import sys

sys.path.append("..")

from util.log import Log


if __name__ == '__main__':
    Log.debug("log-test", "this is debug")
    Log.info("log-test", "this is info")
    Log.warning("log-test", "this is warning")
    Log.error("log-test", "this is error")
    Log.critical("log-test", "this is critical")
