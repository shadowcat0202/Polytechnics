path = "D:/JEON/dataset/drive-download-20220627T050141Z-001/"
filename = ["WIN_20220624_15_58_44_Pro", "WIN_20220624_15_49_03_Pro", "WIN_20220624_15_40_21_Pro",
            "WIN_20220624_15_29_33_Pro"]
idx = 3
print(__doc__)
video = path + filename[idx] + ".mp4"


fopen = []
fopen.append(open(path + filename[idx] + "._은정.txt", "r")) # 실행 전에 은정 _1.txt, 시영 _2.txt 세환 _3.txt 로 변경 해서 해주세요
fopen.append(open(path + filename[idx] + "._세환.txt", "r")) # 실행 전에 은정 _1.txt, 시영 _2.txt 세환 _3.txt 로 변경 해서 해주세요
ff = open(path + filename[idx] + "._final.txt", "w") # 실행 전에 은정 _1.txt, 시영 _2.txt 세환 _3.txt 로 변경 해서 해주세요

while True:
    line1 = fopen[0].readline().split(",")
    line2 = fopen[1].readline().split(",")
    # print(line1, end= ", ")
    # print(line2)

    if not line1 or len(line1) == 1 or len(line2) == 1: break

    txt = f"{line1[0]}, "

    if line1[1] == " 1\n" and line2[1] == " 1\n":
        txt += "1\n"
    else:
        txt += "0\n"

    ff.write(txt)

    # if int(line1[0]) % 100 == 0:
        # print(line1[0])



print("finish")