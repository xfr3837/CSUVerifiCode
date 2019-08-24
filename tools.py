from PIL import Image
import numpy as np
import requests


def label2word(label):
    d = ['2', '3', '4', '6', '7', '8', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'M', 'N', 'P', 'Q', 'R', 'T',
         'U', 'V', 'W', 'X', 'Y', 'Z']
    return d[label]


def word2label(word):
    d = ['2', '3', '4', '6', '7', '8', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'M', 'N', 'P', 'Q', 'R', 'T',
         'U', 'V', 'W', 'X', 'Y', 'Z']
    return d.index(word)


def pre_process(ver_image):
    # 去除黑色噪点
    im = Image.open(ver_image)
    # im.show()
    im_mat = np.asarray(im)  # 40 199 3
    im_mat = np.require(im_mat, requirements=['O', 'W'])
    im_mat.setflags(write=1)
    for i in range(im_mat.shape[0]):
        for j in range(im_mat.shape[1]):
            if im_mat[i][j][0] < 20 and im_mat[i][j][1] < 20 and im_mat[i][j][2] < 20:
                im_mat[i][j] = (255, 255, 255)

    # 灰度化
    im_mat = np.array(Image.fromarray(im_mat).convert('L'), 'f')

    # 二值化
    threshold = 235
    for i in range(im_mat.shape[0]):
        for j in range(im_mat.shape[1]):
            if im_mat[i][j] < threshold:
                im_mat[i][j] = 0
            else:
                im_mat[i][j] = 225

    # 分割
    im = Image.fromarray(im_mat)
    im1 = im.crop((21, 10, 41, 30))
    im2 = im.crop((47, 10, 67, 30))
    im3 = im.crop((73, 10, 93, 30))
    im4 = im.crop((99, 10, 119, 30))
    return im1, im2, im3, im4


def get_image(url, name):
    r = requests.get(url)
    with open('.\\raw_images\\' + name, 'wb') as f:
        f.write(r.content)


if __name__ == '__main__':
    url = 'http://my.csu.edu.cn/cgi-bin/login?method=getLoginVerifiCode'
    total = 100
    j = 0
    for i in range(total):
        name = '%03d.jpg' % i

        get_image(url, name)
        im1, im2, im3, im4 = pre_process('.\\raw_images\\' + name)

        im1.convert("L").save('.\\images\\' + '%04d.bmp' % j)
        j = j + 1
        im2.convert("L").save('.\\images\\' + '%04d.bmp' % j)
        j = j + 1
        im3.convert("L").save('.\\images\\' + '%04d.bmp' % j)
        j = j + 1
        im4.convert("L").save('.\\images\\' + '%04d.bmp' % j)
        j = j + 1

        print('完成 %d' % i)

