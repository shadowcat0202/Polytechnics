def solution1(shirt_size):
    count = [0 for _ in range(6)]
    size_num = {"XS": 0, "S": 1, "M": 2, "L": 3, "XL": 4, "XXL": 5}
    for size in shirt_size:
        count[size_num[size]] += 1
    return count


def solution2(original):
    result = [0 for _ in range(len(original))]
    for i in range(len(original)):
        result[i] = original[len(original) - i - 1]

    return result


def solution3(n):
    if n == 0:
        return 0

    result = 0
    for i in range(n):
        result += 3 * i + 1

    return result


def solution4(original):
    unique_list = list(set(original))
    mmin = 1000
    mmax = 1
    for num in unique_list:
        buf = 0
        for i in original:
            if num == i:
                buf += 1
        if mmax < buf:
            mmax = buf
        if mmin > buf:
            mmin = buf

    return int(mmax / mmin)


if __name__ == "__main__":
    param1 = ["XS", "S", "L","L","XL","S","XS"]
    param2 = [1, 4, 2, 3]
    param3 = 4
    param4 = [1, 3, 3, 1, 3, 3, 2, 1, 999]

    print(solution1(param1))
    print(solution2(param2))
    print(solution3(param3))
    print(solution4(param4))
