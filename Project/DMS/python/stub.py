from collections import deque
q = []
q.append(4)
q.append(5)
q.append(6)
q.pop(0)
print(q)
dq = deque()
for i in range(10):
    q.append(i)
print(q)
q.pop()
print(q)
