def swap():
    # 변수 2개의 값을 서로 바꿔보기
    print("----------------------변수 swap----------------------")
    num1 = 100
    num2 = 200
    print("swap 전 num1:", num1, "num2", num2);
    buf = num2
    num2 = num1
    num1 = buf
    print("swap 후 num1:", num1, "num2", num2);


def valuable_type():
    # print(type(num1))  #해당 변수의 타입 보기
    print("----------------------type 확인----------------------")
    input_num = input("참석자 수를 입력하세요:")  # input()은 String 형태로 저장하기때문에
    cake = input_num * 2
    print("케잌의 수(형번환 전):", cake, "(", type(cake), ")")  # String형인 경우 곱한 횟수만큼 나온다
    cake = int(input_num) * 2  # type(valuable)를 통해서 형변환
    print("케잌의 수(형번환 후):", cake, "(", type(cake), ")")


def input_2_valuable_operator():
    # 2개의 수를 입력받고 합 차를 계산해보자
    print("----------------------수 연산----------------------")
    x = int(input("첫번째 정수"))
    y = int(input("두번째 정수"))
    sumR = f"{x + y}"
    print("더하기", sumR, f"빼기 {x - y}", f"곱하기 {x * y}", f"나누기(실수) {x / y}")
    print("더하기:", x + y)
    print("빼 기:", x - y)
    print("곱하기:", x * y)
    print("나누기(정수):", x // y)  # //(두개면) 정수 return
    print("나누기(실수):", x / y)  # //(한개면) 실수 return
    print("나머지:", x % y)
    print("지수:", x ** y)
    print("반올림:", round(1.278))  # 소수 첫째자리에서 반올림
    print("최소 최대:", min(x, y), ", ", max(x, y))  # 여러개의 파라미터를 넣을 수 있다
    print("절댓값:", abs(-55))


def float_type():
    fa = 125.534
    fb = 21.2475
    fexpA = 3.14E3
    fexpB = 3.14E-3

    print(f'{fa},{fb},{fexpA},{fexpB}')

    octA = 0o1777
    print(octA)

    hexA = 0x1FF24
    print(f'{hexA:d}, {hexA:0x}, {hexA:f}, {hexA:.2f}')

    a = 123
    print(f'{a:0x}, {a:f}')
