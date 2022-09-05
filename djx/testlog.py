import sys
sys.path.append("..")

from util import log

def test_info():
    log.CtxInfo("testlog", "this is info")
    print("end")

def test_error():
    log.CtxError("testlog", "this is error")

def test_info_func():
    log.CtxInfo("func_name", "this is info")

if __name__ == '__main__':
    test_info()
    test_error()
    test_info_func()