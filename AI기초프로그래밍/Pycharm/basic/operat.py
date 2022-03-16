def calc():
    num1 = 3
    num2 = 2
    print(str(num1), "+", str(num2), "=", str(num1+num2))
    print(str(num1), "-", str(num2), "=", str(num1-num2))
    print(str(num1), "*", str(num2), "=", str(num1*num2))
    print(str(num1), "/", str(num2), "=", str(num1/num2))   #실수형
    print(str(num1), "//", str(num2), "=", str(num1//num2)) #정수형
    print(str(num1), "**", str(num2), "=", str(num1**num2))
    print(str(num1), "%", str(num2), "=", str(num1%num2))
    print(str(num1), ">", str(num2), "=", str(num1>num2))
    print(str(num1), ">=", str(num2), "=", str(num1>=num2))
    print(str(num1), "<", str(num2), "=", str(num1<num2))
    print(str(num1), "<=", str(num2), "=", str(num1<=num2))
    print(str(num1), "==", str(num2), "=", str(num1==num2))
    print(str(num1), "!=", str(num2), "=", str(num1!=num2))
    print("not ", str(num1), "!=", str(num2), "=", str(num1//num2))
    print(str(num1), "+", str(num2), "==", str(num2), "=", str(num1//num2))
    print(str(num1), ">", str(num2), " and ", str(num1), "<", str(num2), "=", str(num1>num2 and num1<num2)) # and == &
    print(str(num1), ">", str(num2), " or ", str(num1), "<", str(num2), "=", str(num1>num2 or num1<num2))   # or == |


def simple_formula():
    num1 = 5
    num2 = 2

    num1 += num2
    print(str(num1), "+=", str(num2), "=", str(num1))

    num1 = 5    
    num1 -= num2
    print(str(num1), "-=", str(num2), "=", str(num1))

    num1 = 5
    num1*=num2
    print(str(num1), "*=", str(num2), "=", str(num1))

    num1 = 5
    num1/=num2
    print(str(num1), "/=", str(num2), "=", str(num1))


def number_processing_function():
    print("abs(-1)=",abs(-1)) # 절대값
    print("pow(2,3)=", pow(2, 3)) # 제곱 num1^num2
    print("max(2, 3)=", max(2, 3)) # 최대값
    print("min(2, 3)=",min(2, 3)) # 최소값
    print("round(3.5)=", round(3.5)) # 반올림

    import math
    print("maht.floor(3.88)=", str(math.floor(3.88)))    #올림
    print("maht.ceil(3.14)=", str(math.floor(3.14)))     #내림
    print("maht.sqrt(4)=", str(math.sqrt(4)))            #제곱근
    

def random_processing_function():
    import random
    print("random()=", str(random.random()))    # 0.0 ~ 1.0 미만의 임의의 값 생성
    #int(random() * 10)  # 0~10 미만의 임의의 값 생성
    #int(random() * 10)  # 1~10 이하의 임의의 값 생성
    print("randrange(2, 6)=", str(random.randrange(2, 6)))   # num1 ~ num2 미만의 임의의 값 생성
    #randint(num1, num2) # num1 ~ num2 를 포함한 임의의 값 생성
