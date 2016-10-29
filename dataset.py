import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def jis_to_class(jis_code):
    """ひらがなのJISコードをクラス番号に変換する

    「を」は0、あ〜んまでは1〜45
    46種のひらがなに一意な番号が振られればよかったので番号の並びは関係ない
    """
    # 0xB1は「あ」
    if jis_code >= 0xB1:
        return jis_code - 0xB0
    else:
        # ここは0xA6の「を」だけが通る想定
        return 0


def class_to_one_hot(class_):
    """クラス番号をone-hotな表現に変換する"""
    one_hot = np.zeros(46, np.ubyte)
    one_hot[class_] = 1
    return one_hot


def large_image_to_array(image: Image):
    """画像サイズを縮小して1次元配列を返す

    ETL7は64×63のサイズであり、CNNに食わせるには少し大きい
    とりあえずMNISTのサイズに近い32×32にした
    """
    small_data = image.resize((32, 32))
    small_bytes = small_data.tobytes()
    # MNISTの（CNNの）サンプルに合わせてここでフラットにしているが、別に2次元配列のままでもよかった
    small_array = np.uint8(np.reshape(list(small_bytes), (1024,)))
    return small_array


def read_etl7_binary(input_file, output_file):
    """ETL7のバイナリデータから必要な情報を抜き出す"""
    with open(input_file, 'rb') as file:
        etl7 = file.read()

    data_size = len(etl7)
    index = 0

    data = []
    labels = []
    while index < data_size:
        # 実装したあとに思いだしたけどstructを使ったほうが良かったかも
        # 1文字あたり2052B
        one_character = etl7[index:index + 2052]
        index += 2052
        # ヘッダ等を取り除く
        image_data = one_character[32:2048]
        jis_code = one_character[6]
        if jis_code >= 0xDE:
            # 濁音/半濁音が含まれているが、今回は使わないので無視
            continue
        class_ = jis_to_class(jis_code)
        one_hot_label = class_to_one_hot(class_)
        k = []
        for d in image_data:
            # 1ピクセルあたり4bitなので1Byteを分割する
            k.append(d >> 4)
            k.append(d & 0x0F)
        k = np.array(k, dtype=np.uint8)
        # 用紙の色がノイズとして載っているので平均値+αを0に切り落とす
        # +2は適当なので調整の余地あり？
        k = np.where(k <= k.mean()+2, [0], k)
        k = np.uint8(np.asarray(k.reshape((63, 64))))
        data.append(k)
        labels.append(one_hot_label)
    dataset = {'data': data, 'label': labels}
    with open(output_file, 'wb') as file:
        pickle.dump(dataset, file)


def create_dataset(input_file, output_file, train=False):
    with open(input_file, 'rb') as file:
        large_data_set = pickle.load(file)
    small_images = []
    small_labels = []
    for data, label in zip(large_data_set['data'], large_data_set['label']):
        large_data = Image.fromarray(data)
        # 加工無し
        small_array = large_image_to_array(large_data)
        small_images.append(small_array)
        small_labels.append(label)
        if not train:
            continue
        # 訓練時はランダムに拡大縮小、回転、移動を加えて訓練データを拡張する
        # 1文字あたり9個作るので元データと合わせてデータサイズは10倍になる
        # 実装がいい加減すぎてわかりづらい
        for _ in range(9):
            max_scale = 20  # ±20%
            max_angle = 15  # 20度だと多すぎた
            scale = 1 + (np.random.rand() - 0.5) * max_scale / 100 * 2
            pixel_size = int(64 * scale)
            paste_offset = (64 - pixel_size) // 2
            paste_offset_x = paste_offset + int((np.random.rand() - 0.5) * 64 / 8)
            paste_offset_y = paste_offset + int((np.random.rand() - 0.5) * 64 / 8)
            angle = (np.random.rand() - 0.5) * max_angle * 2
            processed_image = Image.new('L', (64, 63), color=0)
            processed_image.paste(large_data.resize((pixel_size, pixel_size)), (paste_offset_x, paste_offset_y))
            processed_image = processed_image.rotate(angle)
            small_array = large_image_to_array(processed_image)
            small_images.append(small_array)
            small_labels.append(label)

    dataset = {'data': np.array(small_images), 'label': np.array(small_labels)}
    with open(output_file, 'wb') as file:
        pickle.dump(dataset, file)


def display_image():
    """作成したデータを表示する"""
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
        plt.imshow(image, cmap='Greys', clim=(0.0, 15.0))
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.show()


def create_train_data():
    # ETL7のうち4/7を訓練データとする
    read_etl7_binary('data/ETL7LC_1', 'data/data1.pickle')
    create_dataset('data/data1.pickle', 'data/train.pickle', train=True)


def create_test_data():
    # ETL7のうち3/7をテストデータとする
    read_etl7_binary('data/ETL7LC_2', 'data/data2.pickle')
    create_dataset('data/data2.pickle', 'data/test.pickle')

if __name__ == '__main__':
    create_train_data()
    create_test_data()
    display_image()
