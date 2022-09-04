import os

def openMPS(gpu_no, mps_percentage):
    file = os.path.dirname(__file__) + "\mpsopen.sh"
    os.system(("bash %s %s %s" % (file, gpu_no, mps_percentage)))
    print("bash %s" % file)

def closeMPS(gpu_no):
    file = os.path.dirname(__file__) + "\mpsclose.sh"
    os.system(("bash %s %s" % (file, gpu_no)))
    print("bash %s" % file)
    print("exit mps!")