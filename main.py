import matplotlib.pyplot as plt
import tensorflow.keras as k
import pickle
import numpy as np
import json
import math

with open('./xs', 'rb') as f:
    xs = pickle.load(f)

model = k.models.load_model('./model.h5')

model.summary()


print(model.predict(xs).shape)



def save_4d_array_to_image(array, file):
    array = array.T

    w = array.shape[3]
    h = array.shape[2]
    columns = array.shape[1]
    rows = array.shape[0]

    fig = plt.figure(figsize=(40, 40))

    for row in range(rows):
      for column in range(columns):
        img = array[row][column]
        img = np.clip(img, 0, 1)

        fig.add_subplot(rows, columns, column*rows + row + 1)
        plt.imshow(img, cmap='Purples')     

    # plt.show()
    plt.savefig(file, format='svg')
    plt.close()


counter = 0
for l in model.layers:
    wandb = l.get_weights()

    # print(len(wandb))

    if len(wandb) == 2:
        for i in range(2):
            new_shape = list(wandb[i].shape)

            while len(new_shape) < 4:
                new_shape.append(1)

            wandb[i] = wandb[i].reshape(tuple(new_shape))

            save_4d_array_to_image(
                wandb[i], f'./weights_and_biases/{counter}_{i}.svg')

            print(wandb[i].shape)

    counter += 1
