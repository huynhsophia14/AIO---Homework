import numpy as np

def compute_cosine(v1, v2):
    dot_product = np.dot(v1, v2)

    # Compute the norms of the vectors
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Compute the cosine similarity
    cosine_similarity = dot_product/(norm_v1*norm_v2)

    return cosine_similarity

#example
x1 = np.array([1, 2, 3, 4])
x2 = np.array([1, 0, 3, 0])
result = compute_cosine(x1, x2)
print(round(result, 2))
