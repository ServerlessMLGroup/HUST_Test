import os
import sys

print("init mps model...")
path = os.path.dirname(__file__)
# print(path)
os.system("chmod +x " + path + "\mpsclose.sh")
os.system("chmod +x " + path + "\mpsopen.sh")

print("init mps model finishing!")
