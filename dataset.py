import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def jis_to_class(jis_code):
    if jis_code >= 0xB1:
        return jis_code - 0xB0
    else:
        return 0


def class_to_one_hot(class_):
    one_hot = np.zeros(46, np.ubyte)
    one_hot[class_] = 1
    return one_hot


def create_data_set_base():
    with open('data/ETL7LC_2', 'rb') as file:
        etl7 = file.read()

    data_size = len(etl7)
    index = 0

    data = []
    labels = []
    while index < data_size:
        foo = etl7[index:index + 2052]
        index += 2052
        bar = foo[32:2048]
        # data_number = int.from_bytes(foo[0:2], 'big')
        # character_code = foo[2:4]
        # serial_sheet_number = int.from_bytes(foo[4:6], 'big')
        jis_code = foo[6]
        # ebcdic_code = foo[7]
        # eici = int(foo[8])
        # ecg = int(foo[9])
        # male_female_code = int(foo[10])
        # age_of_writer = int(foo[11])
        # serial_data_number = int.from_bytes(foo[12:16], 'big')
        # industry_classification_code = int.from_bytes(foo[16:18], 'big')
        # occupation_classification_code = int.from_bytes(foo[18:20], 'big')
        # if serial_data_number > 46:
        #     continue
        if jis_code >= 0xDE:
            # 濁音/半濁音
            continue
        class_ = jis_to_class(jis_code)
        one_hot_label = class_to_one_hot(class_)
        k = []
        for d in bar:
            k.append(d >> 4)
            k.append(d & 0x0F)
        k = np.array(k, dtype=np.uint8)
        k = np.where(k < 6, [0], k)
        k = np.uint8(np.asarray(k.reshape((63, 64))))
        data.append(k)
        labels.append(one_hot_label)
    dataset = {'data': data, 'label': labels}
    with open('data/data.pickle', 'wb') as file:
        pickle.dump(dataset, file)


def resize_data_set():
    with open('data/data.pickle', 'rb') as file:
        large_data_set = pickle.load(file)
    small_images = []
    for data in large_data_set['data']:
        large_data = Image.fromarray(data)
        small_data = large_data.resize((32, 32))
        small_bytes = small_data.tobytes()
        small_array = np.uint8(np.reshape(list(small_bytes), (1024,)))
        small_images.append(small_array)

    dataset = {'data': np.array(small_images), 'label': np.array(large_data_set['label'])}
    with open('data/test.pickle', 'wb') as file:
        pickle.dump(dataset, file)


def display_image():
    with open('data/train.pickle', 'rb') as file:
        dataset = pickle.load(file)
    images = dataset['data']
    data_size = len(images)
    image_indexes = list(range(data_size))
    np.random.shuffle(image_indexes)
    width = 10
    height = 10
    for i, image_index in enumerate(image_indexes[:width * height]):
        plt.subplot(width, height, i + 1)
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        plt.axis('off')
        image = np.reshape(images[image_index], (32, 32))
        plt.imshow(image, cmap='Greys', clim=(4.0, 15.0))
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.show()


if __name__ == '__main__':
    # create_data_set_base()
    # resize_data_set()
    display_image()
