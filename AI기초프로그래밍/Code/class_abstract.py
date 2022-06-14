# pytorch 기반의 딥러닝 모델을 생성하는 github
# class 상속과 관련해서 간단하게 실습

# class명(부모클래스),
# 부모클래스에 있는 거를 그대로 사용하기 위해 super
# 다중 상속..

class Computer:
    def __init__(self, cpu, ram):
        self.cpu = cpu
        self.ram = ram

    def browse(self):
        print("web 검색")


# Laptop class는 컴퓨터 클래스를 상속 받아 구현
class Laptop(Computer):
    def __init__(self, cpu, ram, battery):
        super().__init__(cpu, ram)
        self.battery = battery

    def move(self):
        print("이동 가능")


class Rect:
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def area(self):
        print("Rect : area()")
        return self.w * self.h

    def perimeter(self):
        return 2 * (self.w + self.h)


class SquareExtends(Rect):
    def __init__(self, w):
        super().__init__(w, w)

    def area(self):
        print("SquareExtends : area()")
        return self.w * self.w

    def perimeter(self):
        return 4 * self.w


class Cube(SquareExtends):
    def surface_area(self):
        # Rect의 area()
        sur_area = super(SquareExtends, self).area()
        return sur_area * 6

    def volumn(self):
        # SquareExtends의 area()
        vol = super().area()
        return vol * self.w


if __name__ == "__main__":
    rect = Rect(2,4)
    print(rect.area(), rect.perimeter())

    sq = SquareExtends(3)
    print(sq.area(), sq.perimeter())

    cube = Cube(3)
    print(cube.surface_area())
    print(cube.volumn())

