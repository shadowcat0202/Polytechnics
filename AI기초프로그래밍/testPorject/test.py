# def add(op="add", *args)  Err *매개변수는 뒤에 오지 못한다

def add(*args, op="add"):
    print(type(args))
    list(args)

    result = 0
    if op == "add":
        result = 0
        for i in args:
            result += i
    elif op == "mul":
        result = 1
        for i in args:
            result *= i
    return result


array = [1, 2, 3, 4, 5]
print(add(1, 2, 3, 4, 5, 6, 4, 8, 9, 41, 89, 6, op="mul"))

add_v = lambda a, b: a + b
print(add_v(3, 4))
