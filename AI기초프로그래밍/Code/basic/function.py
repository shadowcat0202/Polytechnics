"""
Input:*args=입력값이 여러개인 경우
단 *가 붙은 파라미터는 하나만 올 수 있다
"""


def mult_input(*args):
    info = {"add": 0, "mul": 1, "max": args[0], "min": args[0]}
    for i in args:
        info["add"] += i
        info["mul"] *= i
        if i > info["max"]:
            info["max"] = i
        if i < info["min"]:
            info["min"] = i
    return info


def mult_input_v2(a, *args):
    info = {"add": 0, "mul": 1, "max": args[0], "min": args[0]}
    for i in args:
        info["add"] += i
        info["mul"] *= i
        if i > info["max"]:
            info["max"] = i
        if i < info["min"]:
            info["min"] = i
    return info

# 다중 입력은 마지막 파라미터에 와야한다
# 아래와 같이 정의하면 Err
# def mult_input_v3(*args, a):
#     info = {"add": 0, "mul": 1, "max": args[0], "min": args[0]}
#     for i in args:
#         info["add"] += i
#         info["mul"] *= i
#         if i > info["max"]:
#             info["max"] = i
#         if i < info["min"]:
#             info["min"] = i
#     return info


"""
변수=값 형태로 받고 싶은 경우
@:input type(dict)
"""


def func_kwargs(**kwargs):
    print(type(kwargs))
    # 타입이 dict이라 해당 내장 함수도 사용할 수 있다
    print(kwargs.keys())
    print(kwargs.items())
    print(kwargs.values())


"""
변수=값 형태로 받고 싶은 경우
@:return type(tuple)
"""


def kwfunc(**kwargs):
    for item in kwargs.items():
        print(item)

"""
@:return <class:tuple>
"""
def mult_return(a,b):
    return (a+b), (a*b)


def init_fun(name, old, sex=True):
    print(f"이름은 {name}")
    print(f"나이는 {old}")
    if sex:
        print("남자")
    else:
        print("여자")

def test(a):
    a += 1
    print(f"in function {a}")
    