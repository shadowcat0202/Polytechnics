class fourCal:
    def __init__(self):
        self.first = None
        self.second = None
        self.list = []

    def setData(self, first, second, *args):
        self.first = first
        self.second = second
        self.list = args

    def info(self):
        print(f"first={self.first}\nsecond={self.second}\n{self.list}")

    def addAll(self):
        sum = 0
        for i in self.list:
            sum += i
        return sum + self.first + self.second