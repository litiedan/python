import math
import collections
y = 'asdfgasdedcvsf'
result = collections.Counter(y)
print(result.most_common(10))#输出前十项 字符 与 出现次数
print(type(result.most_common(10)))#类型为list
print(result.most_common(1))#输出前一项 字符 与 出现次数
print(type(result.most_common(1)))#类型为list
print(result.most_common(1)[0])#输出list里第一个的tuple
print(type(result.most_common(1)[0]))#类型为tuple
print(result.most_common(1)[0][0])#输出list里的tuple里的第一个元素，即出现次数最多的字符
print(type(result.most_common(1)[0][0]))#类型为str


