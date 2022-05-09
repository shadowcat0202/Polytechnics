
def Try_Except_Final():
    class MyErr(Exception):
        def __str__(self):
            return "MyError!"
    try:
        a = [1, 2]
        if True:
            raise MyErr
        # 4 / 0
    except MyErr as e:
        print(e)
    except ZeroDivisionError as e:
        print(e)


def process_error():
    f = open("./open.txt", "r")

    try:
        a = [1, 2]
        # print(a[3])
        # b = 4 / 0
        f = open("./foo.txt", "r")

    except (ZeroDivisionError, IndexError, FileNotFoundError) as e:
        print(e)
    finally:
        f.close()
