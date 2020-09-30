import os
from os import path

if __name__ == '__main__':
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".cnf"):
                name = file.split(".")[0]
                vtree = "vtrees/%s.vtree" % name
                sdd = "sdds/%s.sdd" % name
                path = os.path.join(root, file)
                with open(path, "r") as input_file:
                    lines = input_file.read().splitlines()
                line = [line for line in lines if len(line) > 0 and line[0] == "p"]
                assert len(line) == 1
                line = line[0]
                numvars = line.split()[2]

                print("Writing graph for file")
                print(sdd, numvars)
                os.system("python write_tf_graph.py %s %s %s" % (vtree, sdd, numvars))
                print("---------------------")


