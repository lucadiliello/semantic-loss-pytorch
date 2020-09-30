import os
from subprocess import check_output

for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".cnf"):
                name = file.split(".")[0]
                g2 = "py2graphs/graph_%s.txt" % name
                g3 = "py3graphs/graph_%s.txt" % name
                exist2 = os.path.isfile(g2)
                exist3 = os.path.isfile(g3)
                if exist2 and exist3:
                    out = check_output("diff %s %s" % (g2, g3), shell=True)
                    out = out.decode("utf-8")
                    if out == "":
                        print("No diff for %s" % name)
                    else:
                        print("Diff for %s" % name)
                        print(out)




