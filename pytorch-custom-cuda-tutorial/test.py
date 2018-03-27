import torch
import mathutils

a = torch.randn(3, 5).cuda()
b = torch.randn(3, 1).cuda()

print(a)
print(b)

print("a += b")
mathutils.broadcast_sum(a, b, *map(int, a.size()))

print(a)
print(b)
