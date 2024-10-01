import re
fname = "C_p_contour"
with open (fname + ".pgf", "r") as f:
    file = f.read()

file = file.replace(fname, r"./04_figures/01_Plots/" + fname)

with open (fname + ".pgf", "w") as f:
    f.write(file)