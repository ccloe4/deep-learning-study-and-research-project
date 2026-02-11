import torch

# x = torch.ones(1, 5)
x = torch.ones(5)
y = torch.zeros(1, 3)

# w, b tensor 에 대한 gradient 계산이 필요하기 때문에 옵션으로 지정
# 또는 선언 후 나중에 w.requires_grad_(True) method 사용 가능
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(1, 3)
b.requires_grad_(True)

z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print(f"Loss: {loss}")

# tensor의 grad_fn property 에 gradient function 저장되어 있음
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# loss 에 대한 w, b의 gradient 계산
# backward(torch.tensor(1.0)) 와 동일
loss.backward()
print(f"Gradient of w: {w.grad}")
print(f"Gradient of b: {b.grad}")

print(z.requires_grad)

# 더 이상 gradient 계산이 필요 없을 때는 with torch.no_grad() 사용
# forward 연산 시 메모리 사용량 절약
with torch.no_grad():
    z = torch.matmul(x, w)+b
    print(f"z: {z}")
    print(f"Gradient function for z = {z.grad_fn}") 
print(z.requires_grad)

# detach() method 사용해도 동일한 효과
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

# output function이 scalar가 아닐 때 actual gradient 가 아니라 Jacobian product 계산함
inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
# 여러 번 backward 호출 시 grad 누적
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond_call\n{inp.grad}")

# grad 초기화 후 다시 backward 호출
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")