# coding:utf-8
import time

def timestring2timestamp(timestring):
    return time.mktime(time.strptime(timestring, '%Y-%m-%d %H:%M:%S'))


def timestamp2timestring(timestamp):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))


def get_last_timestamp(timestamp):
    timestamp = timestamp - 5 * 60
    return timestamp


def get_last_timestr(cretae_time):
    cretae_time = timestring2timestamp(cretae_time)
    last_cretae_time = get_last_timestamp(cretae_time)
    last_cretae_time = timestamp2timestring(last_cretae_time)
    return last_cretae_time


if __name__ == '__main__':
    res = get_last_timestr('2018-08-31 09:29:26')
    print res