import time
from functools import wraps

ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INLINE = list(range(51, 68))
FACE_OUTLINE = list(range(0, 17))
NOTHING = list(range(0, 0))
l = list(i for i in range(68))
lastsave = 0

def counter(func):
    @wraps(func)
    def tmp(lastsave=time.time(), *args, **kwargs):
        print("count 계산중...")
        tmp.count += 1
        time.sleep(0.05)
        print(f"during close: {time.time() - lastsave}")
        if time.time() - lastsave > 5:
            lastsave = time.time()
            tmp.count = 0
        return func(*args, **kwargs)

    tmp.count = 0
    return tmp


@counter
def close():
    print("close()")


check_list = [False, False, False, False, True, True, True, True, True, True, True, True, True, False, False, False,
              False, False, False, ]
for check in check_list:
    if check:
        print("#==================================")
        close()
        print(f'close count : {close.count}')
        if close.count == 15:
            print("Driver is sleeping")
