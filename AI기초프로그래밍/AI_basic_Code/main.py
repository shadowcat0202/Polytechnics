import pandas as pd
import basic.DataType.Dictionary as dic
import basic.DataType.boolean as bol
import basic.DataType.Set as Set


def stub():
    a, b = 1, "python"
    a = [1, 2, 3]
    b = a

    print(a, b, a is b, id(a) == id(b))


if __name__ == '__main__':
    # Set.test()
    # bol.test()
    # stub()
    v = {
        "vehicle": [
            {'name': 'v0', 'x': 1, 'y': 2, 'z': 3, 'h': 4}
        ]
    }
    for i in range(10):
        buf = {
            'name': 'v' + str(i+1),
            'x': i + 1,
            'y': i + 2,
            'z': i + 3,
            'h': i + 4
        }
        v['vehicle'].append(buf)

    print(v)
    for i in range(len(v['vehicle'])):
        print(v['vehicle'][i]['name'])
