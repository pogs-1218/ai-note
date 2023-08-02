import numpy as np

'''
https://numpy.org/doc/stable/user/absolute_beginners.html
'''

def create_vec():
    vec = np.array([1, 2, 3])
    vec_ar = np.arange(4)
    vec_ls = np.linspace(2, 10, 2)

    print(f'vec: {vec}')
    print(f'vec_ar: {vec_ar}')
    print(f'vec_ls: {vec_ls}')

def create_mat():
    mat_2d = np.array([[1, 2, 3], [4, 5, 6]])
    mat_identity = np.identity(3)
   
def main():
    a = np.array([1, 2, 3], dtype='int16')
    b = np.array([[9., 8., 7., 1., 4., 2., 1.], 
                  [6., 5., 4., 7., 2., 3., 1.]], dtype=float)
    print(b)
    print(b.ndim)
    print(b.shape)
    print(a.dtype)
    print(a.itemsize)
    print(a.nbytes)
    print(b.nbytes)

    # Keep in mind!!
    print(b[0, :])
    print(b[:, 1])

    print(b[0, 1:6:2])

    c = np.array([[[1, 2], [3, 4]],
                  [[5, 6], [7, 8]]])
    print(c.ndim)
    print(c.shape)

    print(np.zeros(shape=5))
    print(np.zeros(shape=(2, 2, 2)))
    print(np.full(shape=(2, 2), fill_value=88, dtype='float'))
    print(np.full_like(c, 4))
    print(np.random.rand(3, 2))
    print(np.random.randint(low=0, high=10, size=(2, 2)))
    print(np.identity(3))
    arr = np.array([[1, 2, 3]])
    r1 = np.repeat(arr, 3, axis=1)
    print(r1)

    out = np.ones((5, 5))
    print(out)

    z = np.zeros((3, 3))
    z[1, 1] = 9
    print(z)
    out[1:-1, 1:4] = z
    print(out)

    # Be careful when copying arrays
    a1 = np.array([1, 2, 3])
    b1 = a1
    b1[0] = 100
    print(a1)
    # Not copied! referenced
    b1 = a1.copy()
    b1[0] = 99
    print(a1)
    print(b1)

    print(np.sin(a))

    ### linear algebra
    a = np.ones((2, 3))
    b = np.full((3, 2), 2)
    print(np.matmul(b, a))
    print(a @ b)

    c = np.identity(3)
    np.linalg.det(c) # determinant


    ### Statistics
    stats = np.array([[1, 2, 3], [4, 5, 6]])
    print(np.min(stats))
    print(np.max(stats, axis=0))
    print(np.sum(stats))


    ## Reorgnizing
    before = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    print(before.reshape((8, 1)))

    v1 = np.array([1, 2, 3, 4])
    v2 = np.array([5, 6, 7, 8])
    print(np.vstack([v1, v1, v2]))

    h1 = np.ones((2, 4))
    h2 = np.zeros((2, 2))
    print(np.hstack((h1, h2)))

    ## load data from file
    #data = np.genfromtxt('data.txt', dtype='int16', delimiter=',')

    # boolean masking and advanced indexing
    vv1 = v1 > 2
    print(vv1)
    print(v1[v1 > 2])
    print(np.any(v1 > 2, axis=0))

if __name__ == '__main__':
    main()