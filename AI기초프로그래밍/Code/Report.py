import glob
import json
import os
import pprint


def solution_v1():
    path = "./CAM_FRONT/"
    file_path_list = glob.glob("./CAM_FRONT/*.json")
    for i in range(len(file_path_list)):
        filename = file_path_list[i][len(file_path_list[i]) - file_path_list[i].rfind("\\"):]

        f = open(file_path_list[i])
        json_data = json.load(f)
        # print(json_data)
        result = {"Image": json_data["Image"]}
        # print(result)
        Vehicles = []
        for o in json_data["Object"]:
            # print(o)
            vehicle = {}
            if o["class"] == "Dontcare":
                continue
            if o["level"] == 1 or o["level"] == 2:
                continue
            if o["class"] == "Truck" or o['class'] == "Car":
                vehicle["class"] = "Vehicle"
            else:
                vehicle["class"] = o["class"]

            vehicle["box2d"] = o["box2d"]
            vehicle["level"] = o["level"]
            Vehicles.append(vehicle)

        result["Object"] = Vehicles
        # print(result)
        f.close()

        new_filename = "modified_" + filename
        f = open(path + new_filename, "w")
        json.dump(result, f, indent="\t")
        f.close()


def solution_v2():
    path = "./CAM_FRONT/"
    file_path_list = glob.glob("./CAM_FRONT/*.json")
    for i in range(len(file_path_list)):
        filename = file_path_list[i][len(file_path_list[i]) - file_path_list[i].rfind("\\"):]

        print(f"<<<<<<<< {filename}")
        f = open(file_path_list[i])
        json_data = json.load(f)
        Vehicles = []
        for i, lst in enumerate(json_data["Object"]):
            if lst["class"] == "Dontcare" or lst["level"] == 1 or lst["level"] == 2:
                continue
            Vehicles.append(json_data["Object"][i])
            if lst["class"] == "Truck" or lst["class"] == "Car":
                Vehicles[-1]["class"] = "Vehicle"

        json_data["Object"] = Vehicles
        pprint.pprint(json_data)
        print("===============================================================================")

solution_v2()