
# 변수 temp에 10을 저장하겠다
temp = 90.9
ctemp = (temp - 32.0) * 5.0/9.0
print(ctemp)

num = 10
input_num = input("1~100사이의 숫자 입력하세요.")
print(input_num) #String
input_num_to_int = int(input_num)
if input_num_to_int == num:
    print("맞았습니다.")
else:
    print("틀렸습니다.")