#########################################################
# Cartesian to Polar K-space location mapping in python #
# Author: Dong woo Lee                                  #
# "Generated with the assistance of OpenAI's ChatGPT."  #
#########################################################

import numpy as np
from scipy.sparse import coo_matrix
from scipy.interpolate import griddata

# def grid2(d, k, n):
#     # Ensure input arrays are 1D numpy arrays
#     d = np.array(d).flatten()
#     k = np.array(k).flatten()

#     # Convert k-space samples to matrix indices
#     nx = (n + 1) + 2 * n * k.real
#     ny = (n + 1) + 2 * n * k.imag

#     m = np.zeros((2 * n, 2 * n), dtype=complex)


#     # Loop over samples in kernel
#     for lx in range(-2, 3):
#         for ly in range(-2, 3):
#             # Find nearest samples
#             nxt = np.round(nx + lx).astype(int)
#             nyt = np.round(ny + ly).astype(int)

#             # Separable triangular window
#             kwx = np.maximum(1 - 0.5 * np.abs(nx - nxt), 0)
#             kwy = np.maximum(1 - 0.5 * np.abs(ny - nyt), 0)

#             # If data falls outside matrix, put it at the edge
#             nxt = np.clip(nxt, 1, 2 * n) - 1
#             nyt = np.clip(nyt, 1, 2 * n) - 1

#             # Accumulate gridded data using sparse matrix
#             m_accum = coo_matrix((d * kwx * kwy, (nxt, nyt)), shape=(2 * n, 2 * n)).toarray()
#             m += m_accum

#     # Zero out data at edges
#     m[:, 0] = 0
#     m[:, 2 * n - 1] = 0
#     m[0, :] = 0
#     m[2 * n - 1, :] = 0

#     return m

# 기존 함수 개선
def grid2(d, kx, ky, n, method='linear'):

    d = np.asarray(d).flatten()
    kx = np.asarray(kx).flatten()
    ky = np.asarray(ky).flatten()
    grid_x, grid_y = np.mgrid[min(kx):max(kx):complex(n), min(ky):max(ky):complex(n)]
    grid_d = griddata(points=np.column_stack((kx, ky)), values=d, xi=(grid_x, grid_y), method=method)

    return grid_d, grid_x, grid_y

def cart2pol(x, y): #car to pol
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def generate_polar_k_space(nx, ny):
    radii = np.linspace(0, 1, nx)  # 원의 반경 값 생성
    theta = np.linspace(0, 2 * np.pi, ny)  # 각도 값 생성

    kx_list, ky_list = [], []
    for r in radii:
        kx = r * np.cos(theta)
        ky = r * np.sin(theta)
        kx_list.append(kx)
        ky_list.append(ky)

    kx = np.concatenate(kx_list)
    ky = np.concatenate(ky_list)
    return kx, ky


def create_kspace_locations(nx, ny):
    kx, ky = generate_polar_k_space(nx, ny)
    return kx + 1j*ky  # 복소수 형태로 반환
