def vendingMachine():
    menu = """
    1.커퓌\t1000
    2.카푸취노\t1500
    3.라퉤\t2000
    4.아이스튀\t500
    다른숫자.종료
    """
    inventory = {
        "커퓌":       {"금액":1000,"재고":10},
        "카푸취노":    {"금액":1500,"재고":5},
        "라퉤":        {"금액":2000,"재고":7},
        "아이스튀":     {"금액":500, "재고":10}
    }
    total_price = 0
    
    money = int(input("금액을 투입해 주세요:"))
    while True:
        print(menu)
        print("남은 금액:", money)
        button = int(input("버튼을 눌러주세요"))
        success = False
        if button == 1:
            if inventory['커퓌']["재고"] > 0:  #버튼 1 눌렀고 커퓌 재고 남았을때
                if money >= inventory['커퓌']["금액"]:
                    inventory['커퓌']["재고"] -= 1
                    money -= inventory['커퓌']["금액"] #가격 만큼 까기
                    success = True
                else:
                    print("<!!!!돈이 모자릅니다.!!!!>")
            else:
                print("재고가 없습니다")

        elif button == 2:
            if inventory['카푸취노']["재고"] > 0:  #버튼 1 눌렀고 커퓌 재고 남았을때
                if money >= inventory['카푸취노']["금액"]:
                    inventory['카푸취노']["재고"] -= 1
                    money -= inventory['카푸취노']["금액"] #가격 만큼 까기
                    success = True
                else:
                    print("<!!!!돈이 모자릅니다.!!!!>")
            else:
                print("재고가 없습니다")
                
        elif button == 3:
            if inventory['라퉤']["재고"] > 0:  #버튼 1 눌렀고 커퓌 재고 남았을때
                if money >= inventory['라퉤']["금액"]:
                    inventory['라퉤']["재고"] -= 1
                    money -= inventory['라퉤']["금액"] #가격 만큼 까기
                    success = True
                else:
                    print("<!!!!돈이 모자릅니다.!!!!>")
            else:
                print("재고가 없습니다")

        elif button == 4:
            if inventory['아이스튀']["재고"] > 0:  #버튼 1 눌렀고 커퓌 재고 남았을때
                if money >= inventory['아이스튀']["금액"]:
                    inventory['아이스튀']["재고"] -= 1
                    money -= inventory['아이스튀']["금액"] #가격 만큼 까기
                    success = True
                else:
                    print("<!!!!돈이 모자릅니다.!!!!>")
            else:
                print("재고가 없습니다")

        else:
            print("거래를 종료합니다")
            break

        if success:
            print("정상거래")
