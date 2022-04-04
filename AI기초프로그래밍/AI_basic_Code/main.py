import pandas as pd
import basic.DataType.Dictionary as dic
import basic.DataType.boolean as bol
import basic.DataType.Set as Set


def stub():
    card = True

    if int(input("입력하세요")) > 3000 or card:
        print("Take Taxi")
    else:
        print("nothing")
        
    list_var = [1,2,3,4,5,6,8]
    if 8 in list_var:
        print("8있음")
    else:
        print("8없음")

if __name__ == '__main__':
    stub()