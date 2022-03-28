from datetime import datetime


def 학습데이터처리숙제_3월말까지():
    info = ["전세환", "남", "19950607", 177.0]
    info_t = tuple(info)

    hw1(info)
    hw2(info)
    hw3(info)
    hw4(info)
    hw5(info)


def hw1(info):
    print("""• 정수형, 실수형, 문자열 및 “리스트 내에 리스트” 등을 포함하여 자신을 소개하는 리스트를 만들고,리스트 인덱싱과 슬라이싱 기법 등을 이용하여 자기 소개하는 것 표현해보기""")
    print(f"제 이름은 {info[0]} {info[1]}자 입니다.")
    print(f"나이는 {datetime.today().year - int(info[2][:4]) + 1} 입니다")
    print(f"키는 {info[3]}")
    print()


def hw2(info):
    print("""• 문자열 변수에서 특정 인덱싱으로 접근 후 문자열을 변경할 수 있는가? 답변 및 가능한 방법
        코드 작성 및 제출, mutable, immutable 자료형 정리해보기.""")


def hw3(info):
    print("""• List에서 Append, insert, extend 차이점, remove/pop 차이점, 각각 예제 코드 작성 및 설명""")
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


def hw4(info):
    print("""• 기존 파싱 코드에서 자료형 list와 문자열 관련 함수 split()을 이용하여 동일한 결과 도출하기""")


def hw5(info):
    print("""• 기존 파싱 코드에서 데이터를 어떻게 만들면 더욱 효율적으로 처리할 수 있는지
        – Data 형식 변경 및 추가, 코드 작성 결과 확인 (조건은 동일함, 크기가 100보다 크면 vehicle에서 truck으로    """)
