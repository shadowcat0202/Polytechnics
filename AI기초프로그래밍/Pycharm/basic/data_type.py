def datatype():
    num = 5
    string = "안녕하세요"
    bool_ean = (5 > 3)
    print(str(num), type(num))     #숫자형
    print(string, type(string))    #문자열
    print("5 > 3 = ", type(bool_ean))           #boolean(True, False) 대문자
    print(not bool_ean)             #Java에서의 ! == not

def valuable():
    #파이썬은 data type을 선언하지 않아도 자동으로 class를 판단해 메모리에 알맞게 할당 된다
    animal = "강아지"
    name = "연탄이"
    age = 4
    hobby = "산책"
    is_adult = (age >= 3)

    print("우리집 " + animal + "의 이름은" + name + "이에요")
    print(name + "는 " + str(age) + "살이고, " + hobby + "를 좋아해요") #숫자형을 print에서 사용하려면 str(숫자)를 사용하여 형변환을 해야한다
    print(name + "어른일까요? " + str(is_adult))