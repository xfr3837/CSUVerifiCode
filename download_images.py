import requests

def get_image(url, name):
    r = requests.get(url)
    with open('.\\raw_images\\' + name, 'wb') as f:
        f.write(r.content)


if __name__ == '__main__':
    url = 'http://my.csu.edu.cn/cgi-bin/login?method=getLoginVerifiCode'
    total = 100
    for i in range(total):
        get_image(url, '%03d.jpg' % i)
        print('完成 %d' % i)
