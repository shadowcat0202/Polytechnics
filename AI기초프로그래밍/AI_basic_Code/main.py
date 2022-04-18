import pandas as pd
import basic.DataType.Dictionary as dic
import basic.DataType.boolean as bol
import basic.DataType.Set as Set
import vending_machine as vm
from basic import function as fn
from basic.myClass import fourCal


def d20220218():
    fn.func_kwargs(a=1, b=3)
    fn.kwfunc(x=100, y=200, z="abc")
    print(fn.mult_input(1, 2, 3, 4, 5))
    var_list = [1, 2, 3, 4, 5, 6]
    var_tuple_list = tuple(var_list)
    print(fn.mult_input(*var_list))
    print(fn.mult_input(*var_tuple_list))
    print(fn.mult_input_v2(1, *var_list))
    a = fn.mult_return(1, 2)
    print(f"result={a[0]},{a[1]}")
    print(fn.init_fun("이름이다", 22))
    print("파이썬 어렵다!!!")
    a = 1
    fn.test(1)
    print(f"out function {a}")

    add = lambda *args: [i for i in args]

    print(add(1, 2, 3))
    c1 = fourCal()
    c1.setData(100,2, *[1,2,3,4,5])
    c1.info()
    print(c1.addAll())


if __name__ == '__main__':
    d20220218()
