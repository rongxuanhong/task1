import threading
import urllib.request
import re
import os
import queue


class Download(threading.Thread):
    def __init__(self, que):
        threading.Thread.__init__(self)
        self.que = que

    def run(self):
        while True:
            if not self.que.empty():
                url = self.que.get()
                print('-----正在下载------- {}'.format(url))
                os.system('wget ' + url)
            else:
                break


def startDown(url, rule):
    req = urllib.request.urlopen(url)
    body = req.read().decode('utf8')
    rule = re.compile(rule)
    links = rule.findall(body)
    que = queue.Queue()
    for link in links:
        link = 'https://zenodo.org' + link
        que.put(link)

    for link in range(len(links)):
        d = Download(que)
        d.start()


def main(url, rule):
    """
    程序主入口
    :param url: 下載数据所在的页面url
    :param rule: 匹配下载链接地址的正则式子
    :return: 
    """
    os.chdir('/home/r506/Downloads/')
    startDown(url, rule)


if __name__ == '__main__':
    url = 'https://zenodo.org/record/1228142#.W4ye0M4zaM8'
    rule = '<a class="btn btn-xs btn-default" href=(.*?)><i class="fa fa-download"></i> Download</a>'
    main(url, rule)
