import sys
import re

def calculate_averages(file):
    with open(file) as f:
        f.readline() # title
        f.readline() # compute mode
        avgs = {"total":0,"copying":0,"z-index":0,"sort":0,"b-grid":0,"b'-grid":0,"dens":0,"force":0,"collision":0,"integrate":0,"fps":0}
        count = 0.0
        for line in f:
            line = line.strip()
            fmt1 = re.match(r"(\d+)\.(\d+)sec\s+total:(\d+)ns,\s+copying:(\d+)ns,\s+z-index:(\d+)ns,\s+sort:(\d+)ns,\s+b-grid:(\d+)ns,\s+b'-grid:(\d+)ns,\s+dens:(\d+)ns,\s+force:(\d+)ns,\s+collision:(\d+)ns,\s+integrate:(\d+)ns,\s+FPS:(\d+)\.(\d+)fps", line)
            fmt2 = re.match(r"(\d+)\.(\d+)sec\s+total:(\d+)ns,\s+dens:(\d+)ns,\s+force:(\d+)ns,\s+collision:(\d+)ns,\s+integrate:(\d+)ns,\s+FPS:(\d+)\.(\d+)fps", line)
            if(fmt1):
                sec = float(fmt1.group(1)) + float(fmt1.group(2)) / 100.0
                total = float(fmt1.group(3))
                copying = float(fmt1.group(4))
                zindex = float(fmt1.group(5))
                sort = float(fmt1.group(6))
                bgrid = float(fmt1.group(7))
                bprimegrid = float(fmt1.group(8))
                dens = float(fmt1.group(9))
                force = float(fmt1.group(10))
                collision = float(fmt1.group(11))
                integrate = float(fmt1.group(12))
                fps = float(fmt1.group(13))
                avgs["total"] += total 
                avgs["copying"] += copying 
                avgs["z-index"] += zindex 
                avgs["sort"] += sort 
                avgs["b-grid"] += bgrid 
                avgs["b'-grid"] += bprimegrid
                avgs["dens"] += dens 
                avgs["force"] += force 
                avgs["collision"] += collision 
                avgs["integrate"] += integrate 
                avgs["fps"] += fps 
                count += 1
            elif(fmt2):
                sec = float(fmt2.group(1)) + float(fmt2.group(2)) / 100.0
                total = float(fmt2.group(3))
                dens = float(fmt2.group(4))
                force = float(fmt2.group(5))
                collision = float(fmt2.group(6))
                integrate = float(fmt2.group(7))
                fps = float(fmt2.group(8))
                avgs["total"] += total 
                avgs["dens"] += dens 
                avgs["force"] += force 
                avgs["collision"] += collision 
                avgs["integrate"] += integrate 
                avgs["fps"] += fps 
                count += 1
        for key in avgs:
            avgs[key] = avgs[key] / count
        return avgs


def main():
    benchmark_file = sys.argv[1]
    avgs = calculate_averages(benchmark_file)
    for field in avgs:
        print(field, round(avgs[field], 2), ", ", end="")
    print("\n")
    for field in avgs:
        if(field == "total" or field == "fps"):
            continue
        print(field, round(100* avgs[field]/avgs["total"], 2), ", ", end = "")


if __name__ == "__main__":
    main()