from collections import deque

n = int(input())
sequence = [int(input()) for _ in range(n)]

queue = deque(list(range(1, n+1)))
while queue:
    pass