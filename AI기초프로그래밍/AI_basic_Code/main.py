import pandas as pd
import basic.DataType.Dictionary as dic
import basic.DataType.boolean as bol
import basic.DataType.Set as Set
import vending_machine as vm

def stub():
    tree = 0
    while tree < 10:
        tree += 1
        print(f"나무를 {tree}번 찍었습니다")
        if tree == 10:
            print("넘어감")

    prompt = """
    1.add
    2.del
    3.list
    4.quit
    enter number:
    """
    num = 0
    while num < 4:
        print(prompt)
        num = int(input())
        coff = 3
        while True:
            money = int(input("insert money"))
            if not coff:
                print("run out")
                break
            if money == 300:
                print("coff")
                coff -= 1
            elif money > 300:
                print(f"{money - 300}원 거스름 + 커피")
            else:
                print("돈 모자름")

    a = 0
    while a < 10:
        a += 1
        if a % 2 == 0: continue
        print(a)  # 홀수


if __name__ == '__main__':
    vm.vendingMachine()


