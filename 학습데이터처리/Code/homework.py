from datetime import datetime


def hw1():
    print("Homework1===============================================================================")
    # • 정수형, 실수형, 문자열 및 “리스트 내에 리스트” 등을 포함하여 자신을 소개하는 리스트를 만들고,
    # 리스트 인덱싱과 슬라이싱 기법 등을 이용하여 자기 소개하는 것 표현해보기

    info = ["전세환", "남", "19950607", 177.0]
    print(f"제 이름은 {info[0]} {info[1]}자 입니다.")
    print(f"나이는 {datetime.today().year - int(info[2][:4]) + 1} 입니다")
    print(f"키는 {info[3]}")
    print()


def hw2():
    print("Homework2===============================================================================")
    # • 문자열 변수에서 특정 인덱싱으로 접근 후 문자열을 변경할 수 있는가?
    # 답변 및 가능한 방법 코드 작성 및 제출, mutable, immutable 자료형 정리해보기.

    # Immutable: 숫자(number), 문자열(string), 튜플(tuple)
    # Mutable: 리스트(list), 딕셔너리(dictionary), NumPy의 배열(ndarray)
    x = [1, 2, 3, "String"]
    y = (4, 5, 6, "tuple")
    x[-1] = "Python"
    # y[-1] = "Java"    #Tuples don't support item assignment   #튜플은 값을 수정하지 못한다
    print(x, type(x))
    print(y, type(y))
    print()



def hw3():
    print("Homework3===============================================================================")
    # • List에서 Append, insert, extend 차이점, remove/pop 차이점, 각각 예제 코드 작성 및 설명
    info = ["전세환", "남", "19950607", 177.0]
    buf = info
    print(buf)
    buf.append('newone')
    print(f"buf.append('newone')={buf}")
    print("가장 뒤에 추가\n")

    buf = info
    buf.insert(1, 'newone')
    print(f"buf.insert(1, 'newone')={buf}")
    print("list.insert(a,b) a 인덱스 위치에 b를 삽입\n")

    buf = info
    ext = ["this is new", 123456]
    buf.extend(ext)
    print(f"buf.extend(ext)={buf}")
    print("리스트 끝에 가장 바깥쪽 iterable의 모든 항목을 넣습니다")
    print()


def hw4():
    print("Homework4===============================================================================")
    # • 기존 파싱 코드에서 자료형 list와 문자열 관련 함수 split()을 이용하여 동일한 결과 도출하기
    v = "vehicle 0 0 100 100 vehicle 50 50 200 100"
    v_parse = v.split(" ")
    # print(v_parse)

    test_case = v_parse.count("vehicle")
    count = 0
    for i in range(test_case):
        if int(v_parse[5 * i + 3]) > 100:
            v_parse[5 * i] = "truck"

    v = " ".join(v_parse)
    print(v)
    print()


def hw5():
    print("Homework5===============================================================================")
    # • 기존 파싱 코드에서 데이터를 어떻게 만들면 더욱 효율적으로? 리스트 없이?!?!?! 처리할 수 있는지 –,
    # 코드 작성 결과 확인 (조건은 동일함, 크기가 100보다 크면 vehicle에서 truck으로

    result = ""
    v = "vehicle 0 0 100 100 vehicle 50 50 200 100 vehicle 2 564 99 2 vehicle 12 51 101 352"
    # " vehicle 0 0 100 100 vehicle 50 50 200 100"
    v = " " + v  # 각 묶음별로 동일한 과정을 주기 위해 앞쪽에 빈 공간을 추가

    i = 0
    while i >= 0:
        i = i + 1  # vehicle 시작 지점
        num_value_start_idx = v.find(" ", i) + 1  # 나중에 숫자 비교해서 V or T인지 추가후 숫자부분 넣을 index 위치 기억

        j = num_value_start_idx - 1
        bf = 0  # 이전 인덱스 저장용
        for _ in range(3):  # 3번째 위치의 숫자를 특정하기 위해 3번 돈다 (bf=특정한 숫자 시작지점, j=특정한 숫자 끝나는 지점)
            bf = j + 1
            j = v.find(" ", j + 1)

        if int(v[bf:j]) > 100:  # 대소 비교
            result += "truck "
        else:
            result += "vehicle "

        i = j = v.find(" ", j + 1)
        result += v[num_value_start_idx : j if j >= 0 else len(v)]    # 끝에 도착했을때 j가 find(" ")이기 때문에 마지막에는 -1이므로
        # if j >= 0:
        #     result += v[num_value_start_idx:j]  # 숫자부분 추가
        # else:   # 결론은 끝부분 처리 해주기 위해서
        #     result += v[num_value_start_idx:len(v)]  # j가 find(" ")를 했기 때문에 -1이 나올 가능성이 있다 = 예외처리

        if i != -1:  # 끝나는부분이 아니라면 " "로 이어 적을 준비 아니라면 안적고 끝내기
            result += " "

    print(result)
    print()


if __name__ == "__main__":
    hw1()
    hw2()
    hw3()
    hw4()
    hw5()