def tuple_basic():
    t1 = (1, 2, 3, 4, 5, 6)
    """
    [가능]
    인덱싱
    튜플 더하기 곱하기
    슬라이싱
    len()

    [불가능]
    [!]요소 삭제
    [!]요소 변경
    """

    print(t1[0])    #인덱싱

    t2 = (3, 4)
    print(t1 + t2)  #더하기    (이어 붙이기)
    print(t2 * 3)   #곱하기    (횟수만큼 반복해서 출력)
    print(t1[1:])   #슬라이싱
    print(len(t1))  #길이 구하기
