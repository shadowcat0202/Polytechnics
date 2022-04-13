
v = "vehicle 0 0 100 100 vehicle 50 50 200 100 vehicle 2 564 99 2 vehicle 12 51 101 352"
sum = 0
split_v = v.split()
print(split_v)

for i in range(4):
    if( int(split_v[5*i + 3]) > 100):
        split_v[5*i] = "트럭"
    else:
        split_v[5 * i] = "차량"

    
          

# for j in range(4):
#     if(int(split_v[j+6])>=100):
#         split_v[5] = "truck"

# for j in range(4):
#     if(int(split_v[j+11])>=100):
#         split_v[10] = "truck"

# for j in range(4):
#     if(int(split_v[j+16])>=100):
#         split_v[15] = "truck"

print(split_v)
