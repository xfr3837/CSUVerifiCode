import numpy as np
from keras.engine.saving import load_model

from tools import label2word, pre_process

def predict(im):
    im1, im2, im3, im4 = pre_process(im)
    model = load_model("Veri.h5")
    x = np.empty((4, 20, 20, 1))
    x[0] = ((np.asarray(im1, dtype='float64') / 256).reshape(20, 20, 1))
    x[1] = ((np.asarray(im2, dtype='float64') / 256).reshape(20, 20, 1))
    x[2] = ((np.asarray(im3, dtype='float64') / 256).reshape(20, 20, 1))
    x[3] = ((np.asarray(im4, dtype='float64') / 256).reshape(20, 20, 1))

    pre = ''
    for c in model.predict_classes(x):
        pre = pre + label2word(c)

    return pre


if __name__ == '__main__':
    pre = predict('.\\raw_images\\000.jpg')
    print(pre)
