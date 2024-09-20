import torch
import time

# 生成随机张量
size = 10000
a = torch.rand(size, size)
b = torch.rand(size, size)

# 1. Element-wise Multiplication
start_time = time.time()
c = a * b
end_time = time.time()
print(f"Element-wise multiplication took: {end_time - start_time:.6f} seconds")

# 2. Matrix Multiplication (mm)
start_time = time.time()
c = torch.mm(a, b)
end_time = time.time()
print(f"Matrix multiplication (mm) took: {end_time - start_time:.6f} seconds")

# 3. Matrix Multiplication (matmul)
start_time = time.time()
c = torch.matmul(a, b)
end_time = time.time()
print(f"Matrix multiplication (matmul) took: {end_time - start_time:.6f} seconds")

# 4. Dot Product
a_vec = torch.rand(size)
b_vec = torch.rand(size)
start_time = time.time()
c = torch.dot(a_vec, b_vec)
end_time = time.time()
print(f"Dot product took: {end_time - start_time:.6f} seconds")

# 5. Outer Product
start_time = time.time()
c = torch.ger(a_vec, b_vec)  # or torch.outer(a_vec, b_vec)
end_time = time.time()
print(f"Outer product took: {end_time - start_time:.6f} seconds")

# 6. Batch Matrix Multiplication
batch_size = 10
a_batch = torch.rand(batch_size, size, size)
b_batch = torch.rand(batch_size, size, size)
start_time = time.time()
c = torch.bmm(a_batch, b_batch)
end_time = time.time()
print(f"Batch matrix multiplication took: {end_time - start_time:.6f} seconds")
