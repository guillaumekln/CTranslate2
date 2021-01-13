import ctranslate2
import time
import threading

t1 = ctranslate2.Translator("ende_transformer", device="cuda", device_index=0)
t2 = ctranslate2.Translator("ende_transformer", device="cuda", device_index=1)

print("Sequential")
start = time.time()
t1.translate_file("/data/valid.en", "/tmp/t1.txt", 32)
t2.translate_file("/data/valid.en", "/tmp/t2.txt", 32)
end = time.time()
print(end - start)

print("Parallel")
th1 = threading.Thread(target=t1.translate_file, args=("/data/valid.en", "/tmp/t1.txt", 32))
th2 = threading.Thread(target=t2.translate_file, args=("/data/valid.en", "/tmp/t2.txt", 32))
start = time.time()
th1.start()
th2.start()
th1.join()
th2.join()
end = time.time()
print(end - start)

del t1
del t2
