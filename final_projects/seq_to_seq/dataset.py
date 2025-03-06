import numpy as np

def generate_sequence_dataset(num_samples, max_length):
    X = []
    Y = []
    for _ in range(num_samples):
        length = np.random.randint(1, max_length)
        seq = np.random.randint(1, 10, length)
        X.append(seq)
        Y.append(seq[::-1])  # Target is reverse of source
    return np.array(X), np.array(Y)
