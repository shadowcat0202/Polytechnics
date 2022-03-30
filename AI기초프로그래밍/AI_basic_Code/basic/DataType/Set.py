def test():
    s1 = set([3, 1, 2, 4])
    s2 = set("hello")
    print(s1, s2)

    s1_tuple = tuple(s1)
    print(type(s1_tuple), s1_tuple)
    for item in s1_tuple:
        print(item, end=",")
    print()


    s1 = set([1,2,3,4,5])
    s2 = set([2,3,5])
    print(f"교집합={s1.intersection(s2)}")
    print(f"합집합={s1.union(s2)}")
    print(f"차집합={s1.difference(s2)}")
    print()

    s1 = set([1,3,4])
    s1.add("5")
    print(s1)



