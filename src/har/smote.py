import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np

def euclidian_distance(a,b):

    return np.sqrt(np.sum((a-b)**2))


def smote(data, classes, class_to_generate, k, num_samples):

    # Get data with label equal to label_to_generate

    minority_mask = classes == class_to_generate
    minority_data = data[minority_mask]

    # Generate samples

    synthetic_samples = np.zeros((num_samples,data.shape[1]))

    for i in range(num_samples):

        # random minority sampel

        random_idx = np.random.randint(0, len(minority_data))
        sample = minority_data[random_idx]


        # Calculate distances
        distances = []
        for idx, neighbour in enumerate(minority_data):

            if idx == random_idx:
                continue
            
            dist = euclidian_distance(neighbour, sample)
            distances.append((dist, idx))

        # Determine closest neighbours

        distances.sort(key = lambda x: x[0])
        k_neighbours = []

        for dist, idx  in distances[:k]:
            k_neighbours.append(idx)   


        # Get random NN and calculate syntethic sample

        random_nn_idx = np.random.choice(k_neighbours)
        random_nn = minority_data[random_nn_idx]

        synthetic_value = sample + (np.random.rand() * (random_nn - sample))
        synthetic_samples[i] = synthetic_value

    return synthetic_samples
