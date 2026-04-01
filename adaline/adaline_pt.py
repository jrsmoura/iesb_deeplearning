import torch
import torch.nn as nn
import torch.optim as optim


# ============================
# Modelo AdaLine com PyTorch
# ============================
class AdaLine(nn.Module):
    def __init__(self, n_features: int):
        """
        Modelo AdaLine com uma única camada linear.
        """
        super().__init__()
        self.linear = nn.Linear(n_features, 1)  # Inclui bias automaticamente

    def forward(self, x):
        """
        Forward pass (ativação é identidade).
        """
        return self.linear(x)


# ============================
# Função de treinamento
# ============================
def train(model, X, y, epochs=100, lr=0.01):
    """
    Treina o modelo AdaLine com MSE e SGD.

    :param model: Instância de AdaLine
    :param X: Tensores de entrada (n_samples, n_features)
    :param y: Tensores de saída (n_samples, 1)
    :param epochs: Número de épocas
    :param lr: Taxa de aprendizado
    """
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Época {epoch} | Loss: {loss.item():.4f}")


# github.com/jrsmoura/iesb_deeplearning
# ============================
# Exemplo de uso
# ============================
if __name__ == "__main__":
    # Dados artificiais
    X_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y_data = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

    model = AdaLine(n_features=1)
    train(model, X_data, y_data, epochs=100, lr=0.01)

    # Teste
    with torch.no_grad():
        test = torch.tensor([[5.0]])
        pred = model(test)
        print(f"Predição para 5.0: {pred.item():.4f}")
