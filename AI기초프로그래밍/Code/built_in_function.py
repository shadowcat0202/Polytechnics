import random


def positive(x):
    return x > 0


def Enumerate():
    # index와 같이 사용하고 싶을때
    for i, name in enumerate(["body", "foo", "bar"]):
        print(f"{i}, {name}")


def Filter():
    l = list(random.randint(-5,5) for _ in range(10))
    print(f"origin list:{l}")
    print(f"filter list(nonset):{list(filter(positive, l))}")
    print(f"filter list(set):{set(list(filter(lambda x: x > 0, l)))}")


def Input():
    age_list = []
    while True:
        try:
            age = int(input("나이 입력 (숫자만 0 ~ 99):"))
            if age < 0 or age >= 100:
                break
            print(age)
            age_list.append(age)
        except ValueError:
            print("[warning]:다시 입력. 나이는 숫자만 가능")

    return age_list


def INT():
    # int(x, radix)
    print(int('1A', 16))  # 16진수 -> 10진수


def two_times(x):   return x * 2


def MAP():
    l = [1, 2, 3, 4, 5]
    print(list(map(two_times, l)))
    print(list(map(lambda x: x * 2, l)))


def ZIP():
    # 리스트 가능
    l = [[1, 2, 3], [4, 5, 6]]
    print(list(zip(l[0], l[1])))
    # 문자열도 가능
    s = "abcdefghijklmnop"
    if len(s) % 2 == 0:
        print(list(zip(s[:int(len(s)/2)], s[int(len(s)/2):])))
    else:
        print("문자열이 짝수가 아닙니다")
