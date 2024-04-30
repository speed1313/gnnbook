import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx
import seaborn as sns

# Graph
# 1-2, 1-3, 2-3, 2-4, 2-5

networkx.draw(networkx.Graph([(1, 2), (1, 3), (2, 3), (2, 4), (2, 5)]), with_labels=True)
plt.savefig("graph.png")
plt.show()

# You can change the d, the dimension of the embedding vector of each node(Z[i] in R^{d})
d = 3

# degree matrix
D = jnp.diag(jnp.array([2, 4, 2, 1, 1]))
print("D", D)
# adjacency matrix
W = jnp.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 1, 1],
    [1, 1, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0]
])
print("W", W)


def loss_fn(Z):
    loss = 0
    #for u in range(5):
    #    for v in range(5):
    #        loss += ((D[u,v] + W[u,v]) - Z[u].T @ Z[v])**2
    loss = jnp.sum((D + W - Z @ Z.T)**2)

    return loss


def update(theta, lr=0.01):
    loss, grad = jax.value_and_grad(loss_fn)(theta)
    return theta - lr * grad, loss

Z = jax.random.uniform(key = jax.random.PRNGKey(0), shape=(D.shape[0], d))
for epoch in range(1000):
    Z, loss = update(Z)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss {loss}")

# imshow
plt.imshow(Z)
plt.colorbar()
plt.title(f"Z (d={d})")
plt.savefig(f"Z_d_{d}.png")
plt.show()

inner = Z @ Z.T
sns.heatmap(inner, annot=True)
plt.title("Z @ Z.T")
plt.savefig(f"embedding_similarity_d_{d}.png")
plt.show()

sns.heatmap((D + W), annot=True)
plt.title("D + W")
plt.savefig("D_plus_W.png")
plt.show()


# eigen decomposition
e, V = jnp.linalg.eigh(D + W)
print("eigenvalues", e)
Z = jnp.zeros((D.shape[0], d))
for i in range(min(d, D.shape[0])):
    Z = Z.at[:,i].set(V[:, -1 - i] * jnp.sqrt(e[-1 - i]))

plt.imshow(Z)
plt.colorbar()
plt.title(f"Z (d={d}) SVD")
plt.savefig(f"Z_d_{d}_svd.png")
plt.show()

inner = Z @ Z.T
sns.heatmap(inner, annot=True)
plt.title("Z @ Z.T SVD")
plt.savefig(f"embedding_similarity_d_{d}_svd.png")
plt.show()


