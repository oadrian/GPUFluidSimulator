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
            fmt = re.match(r"(\d+)\.(\d+)sec\s+total:(\d+)ns,\s+copying:(\d+)ns,\s+\s+z-index:(\d+)ns,\s+\s+sort:(\d+)ns,\s+\s+b-grid:(\d+)ns,\s+\s+b'-grid:(\d+)ns,\s+\s+dens:(\d+)ns,\s+\s+force:(\d+)ns,\s+\s+collision:(\d+)ns,\s+\s+integrate:(\d+)ns,\s+\s+FPS:(\d+)\.(\d+)fps", line)
            if(fmt):
                sec = float(fmt.group(1)) + float(fmt.group(2)) / 100.0
                total = float(fmt.group(3))
                copying = float(fmt.group(4))
                zindex = float(fmt.group(5))
                sort = float(fmt.group(6))
                bgrid = float(fmt.group(7))
                bprimegrid = float(fmt.group(8))
                dens = float(fmt.group(9))
                force = float(fmt.group(10))
                collision = float(fmt.group(11))
                integrate = float(fmt.group(12))
                fps = float(fmt.group(13))
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
        for key in avgs:
            avgs[key] = avgs[key] / count
        return avgs


def main():
    benchmark_file = sys.argv[1]
    avgs = calculate_averages(benchmark_file)
    for field in avgs:
        print(field, round(avgs[field], 2), ", ", end="")


if __name__ == "__main__":
    main()