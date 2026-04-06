#* view
# import torch
# t = torch.tensor([[ 1,  2,  3,  4,  5,  6],
#                   [ 7,  8,  9, 10, 11, 12]]) # [2, 6]
# t_view1 = t.view(3, 4)
# print(t_view1)
# t_view2 = t.view(4, 3)
# print(t_view2)

#* transpose
import torch

t1=torch.Tensor([[1,2,3],[4,5,6]]) # [2, 3]
t1=t1.transpose(0,1)
print(t1) # [3, 2]

#* shape 
# 在底层内存实现上，与 view 不同