import re

from neuralparticles.tools.plot_helpers import write_dict_csv
from neuralparticles.tools.param_helpers import getParam, checkUnusedParams

log_path = getParam("log", "")
csv_path = getParam("csv", "")

checkUnusedParams()

if csv_path == "":
    csv_path = log_path[:-4] + ".csv"

p_loss_l = re.compile("[ ]*(\d*)/\d* \[.*\]\s-\sETA")
p_loss = re.compile("(\w*): ([0-9.e-]*)")

history = {}

with open(log_path, 'r') as file:
    for line in file:
        if p_loss_l.match(line):
            for l in p_loss.findall(line[line.find("loss:"):]):
                if not l[0] in history:
                    history[l[0]] = []
                history[l[0]].append(float(l[1]))

write_dict_csv(csv_path, history)