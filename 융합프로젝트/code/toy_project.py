import random
import turtle as t


def start_game():
    t.reset()
    t.penup()
    t.hideturtle()


def show_ans():
    t.reset()


def rand_stamp(_px, _py):
    pass


def draw_line(start_x, start_y, end_x, end_y):
    t.goto(start_x, start_y)
    t.pendown()
    t.goto(end_x, end_y)
    t.penup()


def draw_board(start_x, start_y, pix_size):
    for x in range(start_x, start_x + pix_size[0] * 11, pix_size[0]):
        print('x')
        draw_line(x, start_y, x * 11 + pix_size[0], start_y)
    for y in range(start_y, start_y + pix_size[1] * 11, pix_size[1]):
        print('y')
        draw_line(start_x, y, start_x, y * pix_size[1] + 11)


def draw_turtle_frame():
    num_of_icon = 10
    t.title('toy project')
    t.setup(900, 800)

    for i in range(10):
        t.register_shape(f'./icon/icon_{i}.gif')
        print("register_shape")
    t.penup()
    t.hideturtle()
    t.home()

    start_x_pos = -370
    start_y_pos = 350
    margin_x = 80
    margin_y = 50
    # for line in range(10):
    #     t.goto(start_x_pos, start_y_pos - (margin_y * line))
    #
    #     for i in range(num_of_icon):
    #         icon_num = random.randint(0, num_of_icon - 1)
    #         t.shape(f'./icon/icon_{icon_num}.gif')
    #         t.stamp()
    #         t.forward(margin_x)

    draw_board(start_x_pos, start_y_pos, [margin_x, margin_y])
    t.done()

    # t.register_shape('./icon/icon_cur.gif')   # 저장
    # t.shape('./icon/icon_cur.gif')    # 불러오기
    # t.penup() # 아이콘 들기
    # t.goto(x, y)  # 이동
    # t.stamp() # 찍기

    t.shape()
    t.goto(10, 20)
    print(t.pos())
    t.done()
