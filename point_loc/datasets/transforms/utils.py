import torch
import numpy as np
def skew_symmetric(v):
    """Returns the skew-symmetric matrix of a vector v."""
    return torch.tensor([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def adjoint_matrix(T):
    """
    Computes the 6x6 adjoint representation of a 4x4 transformation matrix T.
    
    Args:
    - T: A 4x4 SE(3) transformation matrix.
    
    Returns:
    - Adj_T: A 6x6 adjoint matrix.
    """
    R = T[:3, :3]  # Rotation part
    t = T[:3, 3]   # Translation part
    t_skew = skew_symmetric(t)
    
    Adj_T = np.block([
        [R, np.zeros((3, 3))],
        [t_skew.numpy() @ R, R]
    ])
    
    return torch.tensor(Adj_T)


def transform_covariance(C_local, T):
    """
    Transforms a covariance matrix C_local from local to global coordinates.
    
    Args:
    - C_local: The local covariance matrix (6x6 or 12x12).
    - T: The 4x4 SE(3) transformation matrix.
    - with_velocities: Boolean indicating if velocities are included.
    
    Returns:
    - C_global: The transformed global covariance matrix.
    """
    if T.shape == (3, 3):
        # covnert to 4*4 transormation with 0 translation
        T = np.block([
            [T, np.zeros((3, 1), dtype=T.dtype)],
            [np.zeros((1, 3), dtype=T.dtype), np.ones((1, 1), dtype=T.dtype)]
        ])
    
    Adj_T = adjoint_matrix(T)           # 6x6 adjoint matrix
    
    # Transform the covariance matrix
    C_global = Adj_T.float() @ C_local.float() @ Adj_T.T.float()
    
    return C_global