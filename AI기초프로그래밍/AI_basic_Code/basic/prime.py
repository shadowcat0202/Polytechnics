<<<<<<< HEAD
import math


def factorization(N):  # 인수분해(약수 구하기)
    arr = []
    while N > 1:
        if N % 2 == 0:
            arr.append(int(N / 2))
            N = N / 2
        else:
            arr.append(int(N / 3))
            N = N / 3

    print("약수의 개수:", len(arr))
    print(list(reversed(arr)))


def divisor_non_sort(N):
    if N == 1:
        print("1개")
        return

    arr = [False for _ in range(N + 1)]
    div_f = []
    div_b = []
    for i in range(1, int(math.sqrt(N)) + 1):
        if not arr[i] and N % i == 0:  # 확인 못한 부분만 골라서 판단
            d = int(N / i)
            div_f.append(i)
            if d != i:
                div_b.append(d)
            arr[i], arr[d] = True, True

    div_r = list(reversed(div_b))
    div_f = div_f + div_r

    print("약수의 개수", len(div_f))
    print(div_f)


def divisor(N):
    if N == 1:
        print("1개")
        return

    arr = [False for _ in range(N + 1)]
    div = []
    for i in range(1, int(math.sqrt(N)) + 1):
        if not arr[i] and N % i == 0:  # 확인 못한 부분만 골라서 판단
            d = int(N / i)
            div.append(i)
            if d != i:
                div.append(d)
            arr[i], arr[d] = True, True

    div.sort()
    print("약수의 개수:", div)
    print(len(div))


def prime_count_improved(N):
    if N == 1:
        print("0개")
        return

    prime_num = []
    for i in range(2, N + 1):
        flag = True
        if i == 1:
            print("0개")
            return
        for j in range(2, int(math.sqrt(i) + 1)):
            if i % j == 0:
                flag = False
                break
        if flag:
            prime_num.append(i)

    print(len(prime_num))


def prime_count(N):
    prime = [True for i in range(N + 1)]
    cnt = 0
    prime[1] = False
    for i in range(2, N + 1):
        size = N // i
        for j in range(2, size + 1):
            if prime[i * j]:
                prime[i * j] = False

    for i in range(2, N):
        if prime[i]:
            cnt += 1
    print(cnt)


def prime():
    testNum = 81
    for i in range(2, testNum):
        # 판별할 수가 특정 수에 나누어 떨어진다면 1을 제외한 수에서 약수가 존재한다는 의미
        if (testNum % i) == 0:
            print(f'{testNum} is not prime num')
            break
        if i == testNum - 1:  # i가 testNum - 1 까지 도착하면 소수라고 판단
            print(f'{testNum} is prime num')

    factorization(testNum)
=======
import math


def factorization(N):  # 인수분해(약수 구하기)
    arr = []
    while N > 1:
        if N % 2 == 0:
            arr.append(int(N / 2))
            N = N / 2
        else:
            arr.append(int(N / 3))
            N = N / 3

    print("약수의 개수:", len(arr))
    print(list(reversed(arr)))


def divisor_non_sort(N):
    if N == 1:
        print("1개")
        return

    arr = [False for _ in range(N + 1)]
    div_f = []
    div_b = []
    for i in range(1, int(math.sqrt(N)) + 1):
        if not arr[i] and N % i == 0:  # 확인 못한 부분만 골라서 판단
            d = int(N / i)
            div_f.append(i)
            if d != i:
                div_b.append(d)
            arr[i], arr[d] = True, True

    div_r = list(reversed(div_b))
    div_f = div_f + div_r

    print("약수의 개수", len(div_f))
    print(div_f)


def divisor(N):
    if N == 1:
        print("1개")
        return

    arr = [False for _ in range(N + 1)]
    div = []
    for i in range(1, int(math.sqrt(N)) + 1):
        if not arr[i] and N % i == 0:  # 확인 못한 부분만 골라서 판단
            d = int(N / i)
            div.append(i)
            if d != i:
                div.append(d)
            arr[i], arr[d] = True, True

    div.sort()
    print("약수의 개수:", div)
    print(len(div))


def prime_count_improved(N):
    if N == 1:
        print("0개")
        return

    prime_num = []
    for i in range(2, N + 1):
        flag = True
        if i == 1:
            print("0개")
            return
        for j in range(2, int(math.sqrt(i) + 1)):
            if i % j == 0:
                flag = False
                break
        if flag:
            prime_num.append(i)

    print(len(prime_num))


def prime_count(N):
    prime = [True for i in range(N + 1)]
    cnt = 0
    prime[1] = False
    for i in range(2, N + 1):
        size = N // i
        for j in range(2, size + 1):
            if prime[i * j]:
                prime[i * j] = False

    for i in range(2, N):
        if prime[i]:
            cnt += 1
    print(cnt)


def prime():
    testNum = 81
    for i in range(2, testNum):
        # 판별할 수가 특정 수에 나누어 떨어진다면 1을 제외한 수에서 약수가 존재한다는 의미
        if (testNum % i) == 0:
            print(f'{testNum} is not prime num')
            break
        if i == testNum - 1:  # i가 testNum - 1 까지 도착하면 소수라고 판단
            print(f'{testNum} is prime num')

    factorization(testNum)
>>>>>>> 367930bd7abb1210c0f4aa3318a60d33fb395f8b
