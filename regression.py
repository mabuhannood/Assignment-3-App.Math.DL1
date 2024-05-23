import torch
from torch import nn


def create_linear_regression_model(input_size, output_size):
    model = nn.Linear(input_size, output_size)
    return model


def train_iteration(X, y, model, loss_fn, optimizer):
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def fit_regression_model(X, y):

    learning_rate = 0.001  
    num_epochs =  10000   
    input_features = X.shape[1]  
    output_features = y.shape[1]  
    model = create_linear_regression_model(input_features, output_features)
    
    loss_fn = nn.MSELoss()  # Use mean squared error loss

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    previous_loss = float("inf")
    tolerance = 1e-8  

    for epoch in range(1, num_epochs + 1):
        loss = train_iteration(X, y, model, loss_fn, optimizer)
        
        # Print the loss every 1000 epochs
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

        # Check if the loss has not changing
        if abs(previous_loss - loss.item()) < tolerance:
            print(f'Training stopped at epoch {epoch} with loss {loss.item()}')
            break
        
        previous_loss = loss.item()
    
    return model, loss

