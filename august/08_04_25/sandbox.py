import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import sklearn
from sklearn.linear_model import LinearRegression

x = np.random.rand(100, 1)
y = 1 + 2 * x + 0.1*np.random.randn(100,1)

idx = np.arange(100)
np.random.shuffle(idx)
train_idx = idx[:80]
val_idx = idx[80:]
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

# numpy linear regression

a = np.random.randn(1)
b = np.random.randn(1)
print(f"Equation: y=2x+1 \nRandom Guess | A: {a}, B: {b}")

lr = 0.01
num_epochs = 2_000
for epoch in range(num_epochs):
    y_hat = a + b * x_train
    error = y_train - y_hat
    loss = (error*2).mean()

    a_derivative = -2 * error.mean()
    b_derivative = -2 * (x_train * error).mean()

    a = a - lr * a_derivative
    b = b - lr * b_derivative

    if epoch % 250 == 0:
        print(f"\tEpoch {epoch}/{num_epochs} | Loss: {loss:.4f}")

print(f"\nAfter Linear Regression | A: {a}, B: {b}")
print(f"Optimal Values | A: 1, B: 2")
linr = LinearRegression()
linr.fit(x_train, y_train)
print("Sanity Check w/ Sci-Kit: ", linr.intercept_, linr.coef_[0])
print(f"Starting Torch Regression...\n\n")



# torch linear regression

device = 'cuda' if torch.cuda.is_available() else 'cpu'
x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)

a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

for epoch in range(num_epochs):
    y_hat = a + b * x_train_tensor
    error = y_train_tensor - y_hat
    loss = (error ** 2).mean()
    loss.backward()
    with torch.no_grad():
        a -= lr * a.grad
        b -= lr * b.grad
    a.grad.zero_()
    b.grad.zero_()

print(f"After Torch Regression | A: {a}, B: {b}")
print(f"Optimal Values | A: 1, B: 2")
print(f"Starting simpler Torch Regression...\n\n")

# simple torch regression

a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.SGD([a, b], lr=lr)

for epoch in range(num_epochs):
    y_hat = a + b * x_train_tensor
    loss = loss_fn(y_train_tensor, y_hat)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(f"Simple Torch Regression | A: {a}, B: {b}")
print(f"Optimal Values | A: 1, B: 2")
print(f"Starting class-based Torch Regression...\n\n")

# torch regression w/ class

a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float, device=device))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float, device=device))

    def forward(self, x):
        return self.a + self.b * x
    
model = LinearRegressionModel().to(device)
print("Initial Guess")
print(model.state_dict())

loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    model.train()
    y_hat = model(x_train_tensor)
    loss = loss_fn(y_train_tensor, y_hat)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(f"Torch Regression w/ class | A: {model.state_dict()['a']}, B: {model.state_dict()['b']}")
print(f"Optimal Values | A: 1, B: 2")
print(f"Starting class-based Torch Regression...\n\n")