import math
# # 设置输入值
# x = -2; y = 5; z = -4

# # 进行前向传播
# q = x + y # q becomes 3
# f = q * z # f becomes -12

# # 进行反向传播:
# # 首先回传到 f = q * z
# dfdz = q # df/dz = q, 所以关于z的梯度是3
# dfdq = z # df/dq = z, 所以关于q的梯度是-4
# # 现在回传到q = x + y
# dfdx = 1.0 * dfdq # dq/dx = 1. 这里的乘法是因为链式法则
# dfdy = 1.0 * dfdq # dq/dy = 1
# print(dfdx)
# print(dfdy)
# print(dfdz)

#######################################################################################################
# w = [2,-3,-3] # 假设一些随机数据和权重
# x = [-1, -2]

# # 前向传播
# dot = w[0]*x[0] + w[1]*x[1] + w[2]
# f = 1.0 / (1 + math.exp(-dot)) # sigmoid函数

# # 对神经元反向传播
# ddot = (1 - f) * f # 点积变量的梯度, 使用sigmoid函数求导
# dx = [w[0] * ddot, w[1] * ddot] # 回传到x
# dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot] # 回传到w
# # 完成！得到输入的梯度
# print(dx)
# print(dw)

#######################################################################################################
# x = 3 # 例子数值
# y = -4

# # 前向传播
# sigy = 1.0 / (1 + math.exp(-y)) # 分子中的sigmoi          #(1)
# num = x + sigy # 分子                                    #(2)
# sigx = 1.0 / (1 + math.exp(-x)) # 分母中的sigmoid         #(3)
# xpy = x + y                                              #(4)
# xpysqr = xpy**2                                          #(5)
# den = sigx + xpysqr # 分母                                #(6)
# invden = 1.0 / den                                       #(7)
# f = num * invden # 搞定！                                 #(8)
# # 回传 f = num * invden
# dnum = invden # 分子的梯度                                         #(8)
# dinvden = num                                                     #(8)
# # 回传 invden = 1.0 / den 
# dden = (-1.0 / (den**2)) * dinvden                                #(7)
# # 回传 den = sigx + xpysqr
# dsigx = (1) * dden                                                #(6)
# dxpysqr = (1) * dden                                              #(6)
# # 回传 xpysqr = xpy**2
# dxpy = (2 * xpy) * dxpysqr                                        #(5)
# # 回传 xpy = x + y
# dx = (1) * dxpy                                                   #(4)
# dy = (1) * dxpy                                                   #(4)
# # 回传 sigx = 1.0 / (1 + math.exp(-x))
# dx += ((1 - sigx) * sigx) * dsigx # Notice += !! See notes below  #(3)
# # 回传 num = x + sigy
# dx += (1) * dnum                                                  #(2)
# dsigy = (1) * dnum                                                #(2)
# # 回传 sigy = 1.0 / (1 + math.exp(-y))
# dy += ((1 - sigy) * sigy) * dsigy                                 #(1)
# # 完成! 嗷~~
# print(dx)
# print(dy)


#######################################################################################################
#初值
x = 3
y = -4
z = 2
w = -1
#前向传播
f1 = x * y
f2 = max(z,w)
f3 = f1 + f2
f4 = f3 * 2
print(f4)
#反向传播
df1x = y #乘法门的局部梯度就是相互交换之后的输入值
df1y = x
df2fz = 1 if z > w else 0 #在取最大值门中，最高值的局部梯度是1.0，其余的是0
df2fw = 1 if w > z else 0 
df3f1 = 1 #加法门的局部梯度都是简单的+1
df3f2 = 1 
df4f3 = 2

df4x = df4f3 * df3f1 * df1x#链式求导
df4y = df4f3 * df3f1 * df1y
print(df4x)
print(df4y)
df4z = df4f3 * df3f2 * df2fz
df4w = df4f3 * df3f2 * df2fw
print(df4z)
print(df4w)

# score = 60

# if score >= 80:
# 	grade = '优秀'
# elif score >= 60:
# 	grade = '及格'
# else:
# 	grade = '不及格'

# print(grade)

# score = 60

# grade = '优秀' if score >= 80 else '及格' if score >= 60 else '不及格'

# print(grade)