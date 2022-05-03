#if문 조건
'''
if 조건1:
    조건1이 참인 경우 실행될 라인
elif 조건2:                          ->없어도 무방
    조건2이 참인 경우 실행될 라인
else:
    위의 조건문들에 해당되지 않는 경우 실행될 라인
'''
def cond():
    import kbhit
    kb = kbhit()
    import turtle
    forward_dis = 100
    #펜의 기능을 t라는 변수에 저장
    t = turtle.Pen()
    while True:
        direction = input("왼쪽, 오른쪽, 종료 -> l, r, q 입력:")
        if kb.kbhit():
            direction = kb.getch()
            c_ord = ord(direction)
            if c_ord == '':
                t.left(60)
                t.forward(forward_dis)
            elif c_ord == "r":
                t.right(60)
                t.forward(forward_dis)
            elif c_ord == "q":
                print("종료")
                break
            else:
                print("입력 문자를 다시 한번 확인 바람")
        turtle.exitonclick()    #화면을 다시 누르기 전까지 유지