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
