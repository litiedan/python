import math
import copy

def V_C(data, rows, cols):
    new_data = []
    for r in range(rows):
        row_data = []
        for c in range(cols):
            row_data.append(data[r * cols + c])
        new_data.append(row_data)
    return new_data

def Full_c(vector, matrix):
    res = []
    for row in matrix:
        tmp = 0
        for i in range(len(vector)):
            tmp += (row[i] * vector[i])
        res.append(tmp)
    return res

def Relu(vector):
    res = []
    for v in vector:
        res.append(v if v > 0 else 0)
    return res

def softmax(vector):
    res = []
    max_v = max(vector)
    for v in vector:
        res.append(math.exp(v - max_v))
    res_sum = sum(res)
    return [r / res_sum for r in res]
def net(x):
    a = Full_c(x, w1)
    r = Relu(a)
    z = Full_c(r, w2)
    y = softmax(z)
    return y


N, M = list(map(int, input().strip().split()))
x = list(map(int, input().strip().split()))
w1 = list(map(float, input().strip().split()))
w1 = V_C(w1, M, N)
w2 = list(map(float, input().strip().split()))
w2 = V_C(w2, 10, M)


y = net(x)

max_val = max(y)
max_idx = y.index(max_val)

new_max_val_record = -10
new_max_idx_record = -1
sensitive_ii = -129

val_of_old_max_idx = 10
no_change_sensitive_ii = -129
for n in range(N):
    x_new = copy.deepcopy(x)
    for ii in range(-128, 128):
        if x[n] == ii:
            continue
        x_new[n] = ii
        y = net(x_new)
        new_max_val = max(y)
        new_max_idx = y.index(new_max_val)
        if (not new_max_idx == max_idx) and new_max_val > new_max_val_record:
            new_max_val_record = new_max_val
            new_max_idx_record = (n + 1)
            sensitive_ii = ii
        new_val_of_old_max_idx = y[max_idx]
        if new_val_of_old_max_idx < val_of_old_max_idx:
            val_of_old_max_idx = new_val_of_old_max_idx
            no_change_sensitive_ii = ii

if not (new_max_idx_record == -1 and sensitive_ii == -129):
    print(new_max_idx_record, sensitive_ii)
else:
    print(max_idx, no_change_sensitive_ii)