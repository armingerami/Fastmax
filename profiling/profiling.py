import time
import torch.nn.functional as F

import math
import torch
from torch import cuda
from torch.autograd import gradcheck
from fastmax import FASTMultiHeadAttention
import fastmax_cuda
import numpy as np
from gla import GatedLinearAttention # from "Gated Linear Attention Transformers with Hardware-Efficient Training"
from flash import FLASH # from "When Linear Attention Meets Autoregressive Decoding: Towards More Effective and Efficient Linearized Large Language Models"

LA_ours = FASTMultiHeadAttention() # ours linear attention implementation
LA_torch = FASTMultiHeadAttention(False) # linear attention implemented using pytorch



# look here
b = 4 # batch
h = 16 # head
d = 128 # dimension per head (i.e. embedding_dimension/h)

# n changes from 10^strt to 10^endd. The number of test points are count
rep = 100
count = 10
strt = 3 # log scale
endd = 5 # log scale
# lengths = [1024,4096,8192]


dtype = torch.float32
print("batch, heads, dimension per head = ",b,",",h,",",d,",")

reg_attention_time = np.zeros(count)
our_LA_time = np.zeros(count)
torch_LA_time = np.zeros(count)
gla_time = np.zeros(count)
flash_time = np.zeros(count)
reg_attention_memory = np.zeros(count)
our_LA_memory = np.zeros(count)
torch_LA_memory = np.zeros(count)
gla_memory = np.zeros(count)
flash_memory = np.zeros(count)
device = torch.device(0)
mask = True


j = -1
print("Our LA Implementation")
for i in np.logspace(strt, endd, count):
# for i in lengths:
    try:
        j += 1
        print(int(i))
        for ii in range(rep):
            # print(ii)
            torch.cuda.empty_cache()
            q = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'),requires_grad=True, dtype=dtype).contiguous()
            k = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'),requires_grad=True, dtype=dtype).contiguous()
            v = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'),requires_grad=True, dtype=dtype).contiguous()
            start_time = time.time()
            e = LA_ours(q,k,v,mask)
            cuda.synchronize()
            end_time = time.time()
            our_LA_time[j] += (end_time - start_time)/rep
        our_LA_memory[j] = torch.cuda.memory_allocated()
        # print(torch.cuda.memory_allocated())
    except:
        print("OOM for token length of ", int(i))

        


print("############################################")
print("Regular Attention")
with torch.backends.cuda.sdp_kernel(
    enable_flash=True, 
    enable_math=False, 
    enable_mem_efficient=True
):
    j = -1
    for i in np.logspace(strt, endd, count):
    #   for i in lengths:
        try:
            j += 1
            print(int(i))
            if(i > 500000): continue
            for ii in range(rep):
                torch.cuda.empty_cache()
                q = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'), dtype=dtype,requires_grad=True)
                k = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'), dtype=dtype,requires_grad=True)
                v = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'), dtype=dtype,requires_grad=True)

                start_time = time.time()
                e = F.scaled_dot_product_attention(q, k, v,is_causal=True)
                cuda.synchronize()
                end_time = time.time()
                reg_attention_time[j] += (end_time - start_time)/rep
            reg_attention_memory[j] = torch.cuda.memory_allocated()
            # print(torch.cuda.memory_allocated())
        except:
            print("OOM for token length of ", int(i))


print("############################################")

j = -1
print("Pytorch LA Implementation")
for i in np.logspace(strt, endd, count):
# for i in lengths:
    try:
        j += 1
        print(int(i))
        for ii in range(rep):
            torch.cuda.empty_cache()
            q = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'), dtype=dtype,requires_grad=True)
            k = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'), dtype=dtype,requires_grad=True)
            v = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'), dtype=dtype,requires_grad=True)
            start_time = time.time()
            e = LA_torch(q,k,v,mask = mask)
            cuda.synchronize()
            end_time = time.time()
            
            torch_LA_time[j] += (end_time - start_time)/rep
        torch_LA_memory[j] = torch.cuda.memory_allocated()
        # print(torch.cuda.memory_allocated())
    except:
        print("OOM for token length of ", int(i))

print("############################################")
print("Gated LA Implementation")
gla = GatedLinearAttention(d = d, h = h, device = torch.device('cuda'))
j = -1
for i in np.logspace(strt, endd, count):
# for i in lengths:
    try:
        j += 1
        print(int(i))
        for ii in range(rep):
            torch.cuda.empty_cache()
            x = torch.normal(0,1,[b,int(i),h*d],device=torch.device('cuda'), dtype=dtype,requires_grad=True)
            start_time = time.time()
            e = gla(x)
            cuda.synchronize()
            end_time = time.time()
            gla_time[j] += (end_time - start_time)/rep
        gla_memory[j] = torch.cuda.memory_allocated()
        # print(torch.cuda.memory_allocated())
    except:
        print("OOM for token length of ", int(i))


print("############################################")
print("Speculative Decoding LA Implementation")
flash = FLASH(dim = h*d, device=torch.device('cuda'), causal = mask)
j = -1
for i in np.logspace(strt, endd, count):
# for i in lengths:
    try:
        j += 1
        print(int(i))
        for ii in range(rep):
            torch.cuda.empty_cache()
            x = torch.normal(0,1,[1,b,int(i),h*d],device=torch.device('cuda'), dtype=dtype,requires_grad=True)
            start_time = time.time()
            e = flash(x)
            cuda.synchronize()
            end_time = time.time()
            flash_time[j] += (end_time - start_time)/rep
        flash_memory[j] = torch.cuda.memory_allocated()
        # print(torch.cuda.memory_allocated())
    except:
        print("OOM for token length of ", int(i))



temp = "["
for i in reg_attention_time: temp += str(i) + ", "
temp += "]"
print("Reg. Att. Time = ", temp)
temp = "["
for i in reg_attention_memory: temp += str(i) + ", "
temp += "]"
print("Reg. Att. Memory = ", temp)
print()

temp = "["
for i in our_LA_time: temp += str(i) + ", "
temp += "]"
print("Our LA Time = ", temp)
temp = "["
for i in our_LA_memory: temp += str(i) + ", "
temp += "]"
print("Our LA Memory = ", temp)
print()

temp = "["
for i in torch_LA_time: temp += str(i) + ", "
temp += "]"
print("Pythorch LA Time = ", temp)
temp = "["
for i in torch_LA_memory: temp += str(i) + ", "
temp += "]"
print("Pytorch LA Memory = ", temp)

print()
temp = "["
for i in gla_time: temp += str(i) + ", "
temp += "]"
print("Gated LA Time = ", temp)
temp = "["
for i in gla_memory: temp += str(i) + ", "
temp += "]"
print("Gated LA Memory = ", temp)

print()
temp = "["
for i in flash_time: temp += str(i) + ", "
temp += "]"
print("Spec. Decod. LA Time = ", temp)
temp = "["
for i in flash_memory: temp += str(i) + ", "
temp += "]"
print("Spec. Decod. LA Memory = ", temp)
