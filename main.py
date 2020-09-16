import matplotlib.pyplot as plt
import tensorflow.keras as k
import pickle
import numpy as np
import json
import math
import tqdm

with open('./xs', 'rb') as f:
    xs = pickle.load(f)

model = k.models.load_model('./model.h5')

model.summary()


def foo(num):
    # a*b=num
    min_perimeter = 10e10
    a = 10e10
    b = 10e10

    for new_a in range(1, num):
        new_b = num/new_a

        if float(new_b).is_integer():
            new_perimeter = new_a+new_b

            if new_perimeter < min_perimeter:
                min_perimeter = new_perimeter
                a = new_a
                b = new_b

    # print(min_perimeter)
    return (int(a), int(b))


def save_4d_array_to_image(array, file):
    w = array.shape[3]
    h = array.shape[2]
    columns = array.shape[1]
    rows = array.shape[0]

    fig = plt.figure(figsize=(40, 40))

    for row in range(rows):
        for column in range(columns):
            img = array[row][column]
            img = np.clip(img, 0, 1)

            total_pixels = img.shape[0]*img.shape[1]

            if total_pixels > 3:
              (x, y) = foo(total_pixels)
              img = img.reshape((x, y))

            fig.add_subplot(rows, columns, column*rows + row + 1)
            plt.imshow(img, cmap='Purples')

    plt.savefig(file, format='svg')
    plt.close()




prediction = np.copy(xs)

new_shape = list(prediction.shape)
while len(new_shape) < 4:
    new_shape.append(1)

(new_shape[1], new_shape[3]) = (new_shape[3], new_shape[1])

prediction = prediction.reshape(tuple(new_shape))

save_4d_array_to_image(
    prediction, f'./internal_results/0.svg')

print(prediction.shape)







counter = 0
for l in model.layers:

    model_copy = k.models.clone_model(model)
    model_copy.set_weights(model.get_weights())

    for _ in range(counter):
        model_copy.pop()

    prediction = model_copy.predict(xs)

    new_shape = list(prediction.shape)
    while len(new_shape) < 4:
        new_shape.append(1)

    (new_shape[1], new_shape[3]) = (new_shape[3], new_shape[1])

    prediction = prediction.reshape(tuple(new_shape))

    save_4d_array_to_image(
        prediction, f'./internal_results/{len(model.layers) - counter}.svg')

    print(prediction.shape)

    counter += 1


counter = 0
for l in model.layers:
    wandb = l.get_weights()

    if len(wandb) == 2:
        for i in range(2):
            new_shape = list(wandb[i].shape)

            while len(new_shape) < 4:
                new_shape.append(1)

            wandb[i] = wandb[i].reshape(tuple(new_shape))

            save_4d_array_to_image(
                wandb[i].T, f'./weights_and_biases/{counter}_{i}.svg')

            print(wandb[i].shape)

    counter += 1
