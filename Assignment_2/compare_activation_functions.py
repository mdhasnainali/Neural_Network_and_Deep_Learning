import cv2
import keras
import time
import csv
import os
import numpy as np

from keras.applications import (
    Xception, VGG16, VGG19, ResNet50V2, ResNet101V2,
    InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2,
    DenseNet121
)
from keras.applications import (
    xception, vgg16, vgg19, resnet_v2, inception_v3,
    inception_resnet_v2, mobilenet, mobilenet_v2,
    densenet
)

models_dict = {
    "Xception": {
        "model": Xception,
        "preprocess": xception.preprocess_input
    },
    "VGG16": {
        "model": VGG16,
        "preprocess": vgg16.preprocess_input
    },
    "VGG19": {
        "model": VGG19,
        "preprocess": vgg19.preprocess_input
    },
    "ResNet50V2": {
        "model": ResNet50V2,
        "preprocess": resnet_v2.preprocess_input
    },
    "ResNet101V2": {
        "model": ResNet101V2,
        "preprocess": resnet_v2.preprocess_input
    },
    "InceptionV3": {
        "model": InceptionV3,
        "preprocess": inception_v3.preprocess_input
    },
    "InceptionResNetV2": {
        "model": InceptionResNetV2,
        "preprocess": inception_resnet_v2.preprocess_input
    },
    "MobileNet": {
        "model": MobileNet,
        "preprocess": mobilenet.preprocess_input
    },
    "MobileNetV2": {
        "model": MobileNetV2,
        "preprocess": mobilenet_v2.preprocess_input
    },
    "DenseNet121": {
        "model": DenseNet121,
        "preprocess": densenet.preprocess_input
    }
}

activation_functions = ['softmax', 'sigmoid', 'relu', 'tanh', 'linear', 'gelu', 'celu', 'elu', 'silu']

# Data Processing
def load_and_generate_ds():
    """Load dataset and filter the dataset for 20 classes
    """
    (train_x, train_y), (test_x, test_y) = keras.datasets.cifar100.load_data()

    target_classes = [
        11,12,13,14,15,16,17,18,19,20,26,27, 28,29,30,51,52,53,54,55 
    ]

    train_y = train_y.flatten()
    test_y = test_y.flatten()
    train_mask = np.where(np.isin(train_y, target_classes), True, False)
    test_mask = np.where(np.isin(test_y, target_classes), True, False)

    filtered_train_x = train_x[train_mask]
    filtered_train_y = train_y[train_mask]
    filtered_test_x = test_x[test_mask]
    filtered_test_y = test_y[test_mask]

    resized_train_x = np.array([cv2.resize(img, (75, 75)) for img in filtered_train_x])
    resized_test_x = np.array([cv2.resize(img, (75, 75)) for img in filtered_test_x])

    scaled_train_y = np.vectorize(lambda y: target_classes.index(y))(filtered_train_y)
    scaled_test_y = np.vectorize(lambda y: target_classes.index(y))(filtered_test_y)


    scaled_train_y = scaled_train_y.reshape(scaled_train_y.shape[0], 1)
    scaled_test_y = scaled_test_y.reshape(scaled_test_y.shape[0], 1)

    return (resized_train_x, scaled_train_y), (resized_test_x, scaled_test_y)


def get_model_memory_usage(model):
    total_params = model.count_params()
    memory = total_params * 4 / (1024 ** 2)  
    return memory


def build_model(model_name, activation_function):
    backbone = models_dict[model_name]["model"](include_top=False, input_shape=(75,75,3))
    backbone.trainable = False
    
    x = keras.layers.Flatten()(backbone.output)
    x = keras.layers.Dense(4096, activation=activation_function)(x)
    x = keras.layers.Dense(4096, activation=activation_function)(x)
    output = keras.layers.Dense(20, activation="softmax")(x)

    model = keras.Model(backbone.input, output)
    return model, models_dict[model_name]["preprocess"]

def train_and_evaluate(model_name, activation_function, train_ds, test_ds):

    model, preprocessing_input = build_model(model_name, activation_function)

    train_ds_x, train_ds_y = train_ds
    test_ds_x, test_ds_y = test_ds
    
    train_ds_x = preprocessing_input(train_ds_x)
    test_ds_x = preprocessing_input(test_ds_x)

    keras.utils.plot_model(model, to_file=f'saved_model_architectures/{model_name}_architecture.png', show_shapes=True, show_layer_names=True, expand_nested=True)
    memory = get_model_memory_usage(model)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.sparse_categorical_crossentropy, metrics=[keras.metrics.SparseCategoricalAccuracy()])

    # Training the Head
    training_start = time.time()
    train_history = model.fit(train_ds_x, train_ds_y, epochs=1, validation_split = 0.1, batch_size = 64)
    training_end = time.time()
    training_time = (training_end - training_start)/100

    evaluation_start = time.time()
    results = model.evaluate(test_ds_x, test_ds_y)
    evaluation_end = time.time()
    evaluation_time = (evaluation_end - evaluation_start)/100

    
    print(f"Model Name: {model_name}")
    print(f"Activation Function: {activation_function}")
    print(f"Training Sample: {train_ds[0].shape[0]}")
    print(f"Testing Sample: {test_ds[0].shape[0]}")
    print(f"Model Size: {memory} MB")
    print(f"Training Time: {training_time} Sec")
    print(f"Evaluation Time: {evaluation_time} Sec")
    print(f"Evaluation Accuracy: {results[1]}")
    print(f"Evaluation Loss: {results[0]}")

    evaluation_data = {
    'Model Name': model_name,
    'Activation Function' : activation_function,
    'Training Samples': train_ds[0].shape[0],
    'Testing Samples': test_ds[0].shape[0],
    'Model Size (MB)': memory,
    'Training Time (Sec)': training_time,
    'Evaluation Time (Sec)': evaluation_time,
    'Evaluation Loss': results[0],
    'Evaluation Accuracy': results[1]
    }

    return evaluation_data


if __name__ == "__main__":
    train_ds, test_ds = load_and_generate_ds()

    # print(train_ds[0].shape)
    # print(train_ds[1].shape)
    # print(test_ds[0].shape)
    # print(test_ds[1].shape)


    for model_name in models_dict.keys():
        for activation_function in activation_functions:
            evaluation_data = train_and_evaluate(model_name, activation_function, train_ds, test_ds)

            csv_file = 'evaluation_metrics.csv'
            file_exists = os.path.isfile(csv_file)
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=evaluation_data.keys())

                if not file_exists:
                    writer.writeheader()
                writer.writerow(evaluation_data)
                    


