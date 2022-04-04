def vendingMachine():
    money = int(input())  # 돈
    menu = """
    1.커퓌
    2.카푸취노
    3.라퉤
    4.아이스튀
    다른숫자.확인
    """
    inventory = {"커퓌": [1000, 10], "카푸취노": [1500, 5], "라퉤": [2000, 7], "아이스튀": [500, 10]}
    inventory["커퓌"] -= 1
    total_price = 0
    while True:
        print(menu)
        button = int(input())

        if button == 1:
            if inventory['커퓌'][1] > 0:
                if money >= inventory['커퓌'][0]:
                    inventory['커퓌'][1] -= 1
                    total_price += inventory['커퓌'][0]
                    money -= inventory['커퓌'][0]
                else:
                    print("돈이 모자릅니다.")
            else:
                print("재고가 없습니다")

            print()
        elif button == 2:
            print()

        elif button == 3:
            print()
        elif button == 4:
            print()
        else:
            break;
