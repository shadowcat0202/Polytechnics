<<<<<<< HEAD
def dictionary_basic():
    a = {1: 'a', 2: 'b'}
    a[4] = 'd'
    print(a, "\n")

    a['name'] = ["Shin", "park", "Kim"]
    print(a, "\n")

    del a[1]
    print("del a[1]")
    a[2] = 'bb'
    a[2] = 'bbb'  # 중복해서 사용하면 가장 최근에 작업했던것만 남기고 나머지는 무시
    print(a, "\n")

    b = {"Dept": ["AI-Engineer", "SMART-Electro"], "StudentNum": [22, 45]}
    print("key:Dept = ", b["Dept"], "\nkey:StudentNum = ", b["StudentNum"], "\n")

    # Err case
    c = {
        # ['name', 'age']: (a['name'], [22, 23, 25]) #List는 hash를 사용 하지 못하기 때문에 불가능 하다
        ('name', 'age'): (a['name'], [22, 23, 25])
    }
    print(c[('name', 'age')][0])

    dic = {"name": "pey", "phone": "0119993323", "birth": "1118"}
    print(f"a.keys()=\t{dic.keys()}")
    print(f"a.values()=\t{dic.values()}")
    print(f"a.items()=\t{dic.items()}")
    print(f"'key' in dic=\t{'phone' in dic}, {'wow' in dic}")
    print(f"dic.get(key)=\t{dic.get('birth')}")
    # get은 존재하지 않는 key사용시 None반환
    # 에러가 나지 않는다면 그냥 인덱스 혹은 key값으로 직접 하는게 프로그램공학적으로는 빠르다
    print('==========================================================')
    a = {
        'name': ["a", "b", "c", "d", "e", "f", "g"],
        'id': [1, 2, 3, 4, 5, 6, 7]
    }
    list_tmp = ['name', 'id']
    a['name'][1] = "K"
    print(a['name'][1])
    print(a)
    dict_keys = a.keys()
    print("for")
    for i in a.keys():
        print(i)
    print(dict_keys)
    print(list_tmp, "\n")

    list_tmp.append('phone-number')
    dict_list = list(a['name'])
    print("dict_list = ", dict_list)    #리스트로 변경해서 사용해야 하는 이유는 원하는 부분을 원하는대로 가공 하기 쉽게 하려고
    dict_list.append('phone-number')
    b = '123'
    c = int(b)
    print(dict_list[0])




=======
def dictionary_basic():
    a = {1: 'a', 2: 'b'}
    a[4] = 'd'
    print(a, "\n")

    a['name'] = ["Shin", "park", "Kim"]
    print(a, "\n")

    del a[1]
    print("del a[1]")
    a[2] = 'bb'
    a[2] = 'bbb'  # 중복해서 사용하면 가장 최근에 작업했던것만 남기고 나머지는 무시
    print(a, "\n")

    b = {"Dept": ["AI-Engineer", "SMART-Electro"], "StudentNum": [22, 45]}
    print("key:Dept = ", b["Dept"], "\nkey:StudentNum = ", b["StudentNum"], "\n")

    # Err case
    c = {
        # ['name', 'age']: (a['name'], [22, 23, 25]) #List는 hash를 사용 하지 못하기 때문에 불가능 하다
        ('name', 'age'): (a['name'], [22, 23, 25])
    }
    print(c[('name', 'age')][0])

    dic = {"name": "pey", "phone": "0119993323", "birth": "1118"}
    print(f"a.keys()=\t{dic.keys()}")
    print(f"a.values()=\t{dic.values()}")
    print(f"a.items()=\t{dic.items()}")
    print(f"'key' in dic=\t{'phone' in dic}, {'wow' in dic}")
    print(f"dic.get(key)=\t{dic.get('birth')}")
    # get은 존재하지 않는 key사용시 None반환
    # 에러가 나지 않는다면 그냥 인덱스 혹은 key값으로 직접 하는게 프로그램공학적으로는 빠르다
    print('==========================================================')
    a = {
        'name': ["a", "b", "c", "d", "e", "f", "g"],
        'id': [1, 2, 3, 4, 5, 6, 7]
    }
    list_tmp = ['name', 'id']
    a['name'][1] = "K"
    print(a['name'][1])
    print(a)
    dict_keys = a.keys()
    print("for")
    for i in a.keys():
        print(i)
    print(dict_keys)
    print(list_tmp, "\n")

    list_tmp.append('phone-number')
    dict_list = list(a['name'])
    print("dict_list = ", dict_list)    #리스트로 변경해서 사용해야 하는 이유는 원하는 부분을 원하는대로 가공 하기 쉽게 하려고
    dict_list.append('phone-number')
    b = '123'
    c = int(b)
    print(dict_list[0])




>>>>>>> 367930bd7abb1210c0f4aa3318a60d33fb395f8b
