import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.utils import add_self_loops, remove_self_loops,add_remaining_self_loops
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


class WeightedSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(WeightedSAGEConv, self).__init__(aggr='mean')  # "Mean" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.size(0)

        # If no edge weights are provided, initialize them to 1
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=x.dtype, device=edge_index.device)

        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value=1, num_nodes=num_nodes)

        # Verify the size of edge_index and edge_weight after adding self-loops
        assert edge_index.size(1) == edge_weight.size(0), f"Size mismatch: edge_index.size(1) = {edge_index.size(1)}, edge_weight.size(0) = {edge_weight.size(0)}"

        # Normalize edge weights
        edge_weight = softmax(edge_weight, edge_index[1], num_nodes=num_nodes)

        # Linear transformation
        x = self.lin(x)

        # Start propagating messages
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        # x_j has shape [E, out_channels]
        return edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        return aggr_out @ self.weight



class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = WeightedSAGEConv(in_channels, hidden_channels)
        self.conv2 = WeightedSAGEConv(hidden_channels, out_channels)
        # self.conv1 = SAGEConv(in_channels, hidden_channels)
        # self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x


def custom_loss(embeddings, edge_index, edge_weight, gamma=1.0, lambda_reg=1e-5):
    # Compute pairwise similarities
    similarity_matrix = torch.matmul(embeddings, embeddings.t())

    # Identity matrix for positive samples
    labels = torch.eye(similarity_matrix.size(0)).to(embeddings.device)

    # Binary Cross-Entropy Loss for contrastive learning
    bce_loss = F.binary_cross_entropy_with_logits(similarity_matrix, labels)

    # Modularity term
    m = edge_weight.sum()
    if m == 0:
        modularity_loss = 0
    else:
        modularity = 0
        for i in range(edge_index.size(1)):
            u, v = edge_index[0, i], edge_index[1, i]
            modularity += edge_weight[i] * (similarity_matrix[u, v] - (edge_weight[u] * edge_weight[v]) / (2 * m))

        modularity_loss = -gamma * modularity / m

    # L2 Regularization term
    l2_reg = lambda_reg * torch.norm(embeddings, 2)

    # Combined loss
    loss = bce_loss + modularity_loss + l2_reg
    return loss

def train_unsupervised(data, model, optimizer, epochs=200, gamma=1.0):
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            embeddings = model(data)
            # Contrastive loss for unsupervised learning: Calculates the pairwise similarities between node embeddings.
            # loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(embeddings, embeddings.t()))))
            # Compute the custom loss
            loss = custom_loss(embeddings, data.edge_index, data.edge_weight, gamma=gamma)
            # loss = contrastive_loss(embeddings)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')
        return model


def contrastive_loss(embeddings):
    # Compute pairwise similarities
    similarity_matrix = torch.matmul(embeddings, embeddings.t())
    labels = torch.eye(similarity_matrix.size(0)).to(embeddings.device)

    # Binary Cross-Entropy Loss for contrastive learning
    bce_loss = F.binary_cross_entropy_with_logits(similarity_matrix, labels)
    return bce_loss
def cluster_embeddings(data, model, num_clusters):
    model.eval()
    with torch.no_grad():
        embeddings = model(data)
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(embeddings.cpu().numpy())
    return clusters


def estimate_num_clusters_silhouette(embeddings, max_clusters):
    similarity_matrix = calculate_similarity_matrix(embeddings)

    silhouette_scores = []

    for k in range(2, max_clusters):
        spectral = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='kmeans')
        try:
            clusters = spectral.fit_predict(similarity_matrix)
            score = silhouette_score(similarity_matrix, clusters, metric='precomputed')
            silhouette_scores.append(score)
        except Exception as e:
            print(f"Error for k={k}: {e}")

    # Find the number of clusters with the highest silhouette score
    optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 because the range starts at 2
    return optimal_clusters


def cluster_embeddings_spectral(embeddings,max_cluster, num_clusters=None):
    if num_clusters is None:
        num_clusters = estimate_num_clusters_silhouette(embeddings,max_cluster)

    similarity_matrix=calculate_similarity_matrix(embeddings)


    spectral = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', assign_labels='kmeans')
    clusters = spectral.fit_predict(similarity_matrix)

    return clusters

def evaluate_clustering(similarity_matrix, clusters):
    # Calculate silhouette score
    silhouette_avg = silhouette_score(similarity_matrix, clusters, metric='precomputed')
    print(f'Silhouette Score: {silhouette_avg}')

    # Calculate Davies-Bouldin index
    davies_bouldin_avg = davies_bouldin_score(similarity_matrix, clusters)
    print(f'Davies-Bouldin Index: {davies_bouldin_avg}')

    # Calculate Calinski-Harabasz index
    calinski_harabasz_avg = calinski_harabasz_score(similarity_matrix, clusters)
    print(f'Calinski-Harabasz Index: {calinski_harabasz_avg}')


def calculate_similarity_matrix(embeddings):
    similarity_matrix = torch.matmul(embeddings, embeddings.t()).cpu().numpy()
    np.fill_diagonal(similarity_matrix, 0)  # Ensure diagonal elements are zero
    # Normalize the similarity matrix
    similarity_matrix = (similarity_matrix - similarity_matrix.min()) / (
            similarity_matrix.max() - similarity_matrix.min())
    # Standardize the similarity matrix
    similarity_matrix = (similarity_matrix - similarity_matrix.mean()) / similarity_matrix.std()
    # Ensure non-negative values
    similarity_matrix[similarity_matrix < 0] = 0
    # Check for NaNs or Infs and handle them
    if np.isnan(similarity_matrix).any() or np.isinf(similarity_matrix).any():
        similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0,
                                          posinf=np.max(similarity_matrix[np.isfinite(similarity_matrix)]),
                                          neginf=np.min(similarity_matrix[np.isfinite(similarity_matrix)]))
    # Set diagonal elements to zero
    np.fill_diagonal(similarity_matrix, 0)
    return similarity_matrix