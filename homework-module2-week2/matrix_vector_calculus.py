import numpy as np

def compute_vector_length(vector):
    len_of_vector = np.linalg.norm(vector)
    return len_of_vector


def compute_dot_product(vector1, vector2):
    result = np.dot(vector1, vector2)
    return result


def matrix_multi_vector(matrix, vector):
    result = matrix@vector
    return result


def matrix_multi_matrix(matrix1, matrix2):
    result = matrix1@matrix2
    len_of_vector = result.shape

    return len_of_vector


def inverse_matrix(matrix):
    result = np.linalg.inv(matrix)

    return result


# compute_vector_length
vector = np.array([-2, 4, 9, 21])
result = compute_vector_length([vector])
print(round(result, 2))
print(compute_vector_length)

# vector & vector dot product
v1 = np.array([0, 1, -1, 2])
v2 = np.array([2, 5, 1, 0])
result = compute_dot_product(v1, v2)
print(round(result, 2))

# matrix & vector dot product
x = np.array([[1, 2],
              [3, 4]])
k = np.array([1, 2])
result_2 = matrix_multi_vector(x, k)
print(result_2)

# matrix & matrix dot product
x1 = np.array([[1, 2],
              [3, 4]])
x2 = np.array([[-3, 2],
               [4, 1]])
result_3 = matrix_multi_matrix(x1, x2)
print(result_3)

#inverse matrix
x = np.array([[-2, 6],
              [8, 4]])
result_4 = inverse_matrix(x)
print(result_4)
