import numpy as np

def test1b_H(H):
    H_correct = np.eye(25)
    if np.isfinite(H).all() and np.array_equal(H, H_correct):
        print("H mask matrix is correct")
    else:
        print("H mask matrix is incorrect")

def test1b_H_alt(H_alt):
    H_correct = np.eye(25)
    H_alt_correct = np.vstack([H_correct[::2], H_correct[1::2]])
    if np.isfinite(H_alt).all() and np.array_equal(H_alt, H_alt_correct):
        print("H_alt mask matrix is correct")
    else:
        print("H_alt mask matrix is incorrect")

def test2(H, H_alt):
    H_correct = np.eye(1024)
    H_alt_correct = np.vstack([H_correct[::2], H_correct[1::2]])
    if np.isfinite(H).all() and np.array_equal(H, H_correct):
        print("H mask matrix is correct")
    else:
        print("H mask matrix is incorrect")
    if np.isfinite(H_alt).all() and np.array_equal(H_alt, H_alt_correct):
        print("H_alt mask matrix is correct")
    else:
        print("H_alt mask matrix is incorrect")

