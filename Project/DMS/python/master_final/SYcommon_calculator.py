def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (0.5)


def get_width_and_height(landmarks):
    xy_ldmks = [ loc for loc in landmarks]

    print(xy_ldmks)


    # return (wth, hgt)
#
# def get_width_and_height(landmark4Left, landmark4Top, landmark4Right, landmark4Bottom):
#     wth = distance(landmark4Left, landmark4Right)
#     hgt = distance(landmark4Top, landmark4Bottom)
#
#     return (wth, hgt)
