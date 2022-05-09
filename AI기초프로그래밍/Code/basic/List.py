<<<<<<< HEAD
def List_basic():
    # 리스트
    # valuable = [value1, value2, ...] #물론 String형은 'value1'와 같은 방법으로 입력
    # 숫자도 가능
    List = [10, 20, 30]
    List2 = [10, 20, 30]
    List3 = [1, 2, 0.1, "String"]

    print(id(List), id(List2), id(List) == id(List2))
    print(type(list), List)
    print(List3)
    # 문자열도 가능
    List = ["홍길동", "바둑이", "영희"]
    print(type(list), List, "\n")
    # 혼합도 가능
    List = [10, "홍길동", '가']
    print(type(list), List)

    # <class:list>.index(variable)   -> return index
    print("\"홍길동\"의 위치(index)는 ", List.index("홍길동"), "\n")

    # <class:list>.append(variable)  -> list의 뒤에 추가
    List.append("append")
    #List = List + ["append"]   #동일
    print(List, "\n")

    # <class:list>.insert(index, variable)   ->list의 index위치에 variable을 삽입
    List.insert(1, "insert")
    print(List, "\n")

    # <class:list>.pop() -> 뒤에서 한개씩 꺼냄
    print("pop '", List.pop(), "'")
    print(List, "\n")

    # <class:list>.count(variable)  -> variable의 개수
    List = [1, 2, 3, 1]
    print("List.count(1)=", List.count(1))

    room_list = [38, 3, ["study", "bed", "game"], "1층", "3억"]
    print(room_list, "\n")


def List_slicing():
    a = [1, 2, 3, 4, 5, 6]
    print(f'a = {a}')
    print("a[0:2]=", a[0:2])
    print("a[:3]=", a[:3])
    print("a[4:]=", a[4:])

    b = [7, 8, 9]
    print(f'\nb = {b}')
    print(f'a+b = {a + b}')

    a[2] = 10
    print("a[2] = \t\t\t\t\t", a)
    del a[1]

    print("del a[1] = \t\t\t\t", a)

    a.append([100, 101])
    print("a.append([100, 101])= \t", a)

    del a[-1]
    print("del a[-1]=\t\t\t\t", a)

    a.sort()
    print("a.sort() =\t\t\t\t", a)

    a.reverse()
    print("a.reverse() =\t\t\t", a)

    a.insert(3, 50)
    print("a.insert(3, 50) =\t\t", a)

    a.remove(6)
    print("a.remove(6) =\t\t\t", a)

    a.extend(b)  # a += b
    print("a.extend(b) =\t\t\t", a)


=======
def List_basic():
    # 리스트
    # valuable = [value1, value2, ...] #물론 String형은 'value1'와 같은 방법으로 입력
    # 숫자도 가능
    List = [10, 20, 30]
    List2 = [10, 20, 30]
    List3 = [1, 2, 0.1, "String"]

    print(id(List), id(List2), id(List) == id(List2))
    print(type(list), List)
    print(List3)
    # 문자열도 가능
    List = ["홍길동", "바둑이", "영희"]
    print(type(list), List, "\n")
    # 혼합도 가능
    List = [10, "홍길동", '가']
    print(type(list), List)

    # <class:list>.index(variable)   -> return index
    print("\"홍길동\"의 위치(index)는 ", List.index("홍길동"), "\n")

    # <class:list>.append(variable)  -> list의 뒤에 추가
    List.append("append")
    #List = List + ["append"]   #동일
    print(List, "\n")

    # <class:list>.insert(index, variable)   ->list의 index위치에 variable을 삽입
    List.insert(1, "insert")
    print(List, "\n")

    # <class:list>.pop() -> 뒤에서 한개씩 꺼냄
    print("pop '", List.pop(), "'")
    print(List, "\n")

    # <class:list>.count(variable)  -> variable의 개수
    List = [1, 2, 3, 1]
    print("List.count(1)=", List.count(1))

    room_list = [38, 3, ["study", "bed", "game"], "1층", "3억"]
    print(room_list, "\n")


def List_slicing():
    a = [1, 2, 3, 4, 5, 6]
    print(f'a = {a}')
    print("a[0:2]=", a[0:2])
    print("a[:3]=", a[:3])
    print("a[4:]=", a[4:])

    b = [7, 8, 9]
    print(f'\nb = {b}')
    print(f'a+b = {a + b}')

    a[2] = 10
    print("a[2] = \t\t\t\t\t", a)
    del a[1]

    print("del a[1] = \t\t\t\t", a)

    a.append([100, 101])
    print("a.append([100, 101])= \t", a)

    del a[-1]
    print("del a[-1]=\t\t\t\t", a)

    a.sort()
    print("a.sort() =\t\t\t\t", a)

    a.reverse()
    print("a.reverse() =\t\t\t", a)

    a.insert(3, 50)
    print("a.insert(3, 50) =\t\t", a)

    a.remove(6)
    print("a.remove(6) =\t\t\t", a)

    a.extend(b)  # a += b
    print("a.extend(b) =\t\t\t", a)


>>>>>>> 367930bd7abb1210c0f4aa3318a60d33fb395f8b
