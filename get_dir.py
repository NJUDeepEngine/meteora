import os

path = "/data0/ljy/workspace/BIG-bench/fuze_15/"
print(os.listdir(path), len(os.listdir(path)))
print([name for name in os.listdir(path) if name != "alpaca"])