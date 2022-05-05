def read_value(file_name):
    rtn_list = []
    try:
        f = open(file_name, 'r')
        while True:
            line = f.readline()
            if not line:    break
            rtn_list.append(list(map(float,
                                     line.rstrip("\n").split("\t"))))
    except FileNotFoundError as e:
        print(e)
    finally:
        f.close()

    return rtn_list