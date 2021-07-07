# 简单理解，多进程就是同时运行多个.py文件已达到并行多任务
# 多线程就是一个py文件下可以并行多个任务

# Python中使用线程有两种方式：函数或者用类来包装线程对象。

# 函数式：调用 _thread 模块中的start_new_thread()函数来产生新线程。语法如下:
# _thread.start_new_thread ( function, args[, kwargs] )

#     function - 线程函数。
#     args - 传递给线程函数的参数,他必须是个tuple类型。
#     kwargs - 可选参数。

import _thread
import time

# 为线程定义一个函数
def print_time( threadName, delay):
   count = 0
   while count < 5:
      time.sleep(delay)
      count += 1
      print ("%s: %s" % ( threadName, time.ctime(time.time()) ))
def func01(threadName):
    a = 0
    while True:
        a = a + 1
        if(a%5000000 == 0):
            print(a)
# 创建两个线程
try:
   _thread.start_new_thread( print_time, ("Thread-1", 2, ) )
   _thread.start_new_thread( print_time, ("Thread-2", 4, ) )
   _thread.start_new_thread( func01,("Thread-3",))#同时运行两个for循环
   _thread.start_new_thread( func01,("Thread-4",))
except:
   print ("Error: 无法启动线程")

while 1:
   pass