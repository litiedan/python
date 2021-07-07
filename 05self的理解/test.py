class A:
    def __init__(self):#相当于构造函数，创建类时默认执行
        print("aaaaaa")
    def func(self):
        print(self)                      #指向的是类的实例
        print(self.__class__)         #指向的是类

a = A()
a.func()
#<__main__.A object at 0x02C40F10>
#<class '__main__.A'>

#a=A() a.func()过程等价于
# A.func(a)
