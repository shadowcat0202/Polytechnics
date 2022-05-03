def vendingMachine():
    menu = """
    1.커퓌\t1000
    2.카푸취노\t1500
    3.라퉤\t2000
    4.아이스튀\t500
    다른숫자.종료
    """
    inventory = {
        "커퓌": {"금액": 1000, "재고": 10},
        "카푸취노": {"금액": 1500, "재고": 5},
        "라퉤": {"금액": 2000, "재고": 7},
        "아이스튀": {"금액": 500, "재고": 10}
    }
    total_price = 0

    money = int(input("금액을 투입해 주세요:"))
    while True:
        print(menu)
        print("남은 금액:", money)
        button = int(input("버튼을 눌러주세요"))

        if button == 1:
            name = "커퓌"
        elif button == 2:
            name = "카푸취노"
        elif button == 3:
            name = "라퉤"
        elif button == 4:
            name = "아이스튀"
        else:
            print("거래를 종료합니다.")

        if inventory[name]["재고"] > 0:  # 재고 남았을때
            if money >= inventory[name]["금액"]:
                inventory[name]["재고"] -= 1
                money -= inventory[name]["금액"]  # 가격 만큼 까기
                print("정상거래")
            else:
                print("<!!!!돈이 모자릅니다.!!!!>")
        else:
            print("<!!!!재고가 없습니다!!!!>")
