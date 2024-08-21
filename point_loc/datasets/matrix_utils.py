import torch


def vector_to_upper_triangular_matrix(vector, dim=6):
    """
    Convert a vector of size (N, 21) to a symmetric matrix of size (N, 6, 6).
    The vector represents the upper triangular elements of the matrix.
    """
    N = vector.shape[0]
    
    # Create indices for the upper triangular part
    indices = torch.triu_indices(dim, dim)
    
    # Create an empty matrix
    matrix = torch.zeros(N, dim, dim, device=vector.device)
    
    # Fill the upper triangular part using advanced indexing
    matrix[:, indices[0], indices[1]] = vector
    
    # Make the matrix symmetric
    return matrix 

def cholesky_undecomposition(upper_matrix: torch.Tensor) -> torch.Tensor:
    """
    Perform inverse Cholesky decomposition on a upper triangular matrix to
    get a symmetric matrix.
    Args:
        upper_matrix (torch.Tensor): Input upper triangular matrix of size Batch, D, D.
    """
    symmetric_matrix = torch.bmm(upper_matrix.transpose(-2, -1), upper_matrix)
    return symmetric_matrix
    

def symetric_matrix_to_upper_triangular_vector(symmetric_matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert a symmetric matrix to a vector representing the upper triangular part.
    Args:
        symmetric_matrix (torch.Tensor): Input symmetric matrix of size Batch, D, D.
    """
    
    # Create indices for the upper triangular part
    indices = torch.triu_indices(symmetric_matrix.size(-1), symmetric_matrix.size(-1))
    
    # Extract the upper triangular elements using advanced indexing
    vector = symmetric_matrix[:, indices[0], indices[1]]
    assert len(vector.shape) == 2
    return vector
    
def vector_to_symmetric_matrix(vector: torch.Tensor, dim=6) -> torch.Tensor:
    """
    Convert a vector of size (N, 21) to a symmetric matrix of size (N, 6, 6).
    The vector represents the upper triangular elements of the matrix.
    """
    if len(vector.shape) == 1:
        N = 1
    else:
        N = vector.shape[0]
    
    # Create indices for the upper triangular part
    indices = torch.triu_indices(dim, dim)
    
    # Create an empty matrix
    matrix = torch.zeros(N, dim, dim, device=vector.device)
    
    # Fill the upper triangular part using advanced indexing
    matrix[:, indices[0], indices[1]] = vector
    
    # Make the matrix symmetric
    matrix = matrix + matrix.transpose(-2, -1) - torch.diag_embed(torch.diagonal(matrix, dim1=-2, dim2=-1))
    return matrix.squeeze()