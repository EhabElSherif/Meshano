# This is a sample Python script.
import subprocess
import os

#TODO install subprocess through pip, command is: pip install subprocess.run
#TODO install meshio: pip install meshio[all]

def handleInput(inputFileName):
    # Use a breakpoint in the code line below to debug your script.
    with open(inputFileName) as f:
        if os.path.isfile('pointCloud.txt'):
            os.remove("pointCloud.txt")
        with open("pointCloud.txt", "a") as o:
            for _ in range(13):
                next(f)
            for line in f:
                # read only 6 numbers in each line
                print(' '.join(line.split()[0:6]), file=o)
    args = "mesh.exe --in pointCloud.txt --out output_%s" % inputFileName
    subprocess.call(args, shell=False)
    x = "pleaseWork"
    args2 = "meshio-convert    "+"output_"+inputFileName[:-4]+".ply "+inputFileName[:-4]+".obj"
    print(args2)
    subprocess.call(args2, shell=False)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    handleInput("shalaby.ply")