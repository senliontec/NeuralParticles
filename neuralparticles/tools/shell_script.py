import os
from subprocess import Popen, PIPE

def manta_script(manta_path, path):
    return "%s/manta %s%s" % (manta_path, path)

def run_manta(manta_path, scene, param={},verbose=False,logfile=""):
    command = [manta_path+"manta", manta_path + "../" + scene]

    for k, v in param.items():
        command += [k, str(v)]
        
    if logfile != "":
        command += [">",logfile]
        command += ["; echo", "output written into %s" % logfile]
        
    print(" ".join(command) + "\n")

    proc = Popen(command, stdin=None, stdout=PIPE, stderr=PIPE)

    for line in proc.stdout:
        if verbose:
            print(line.decode('utf-8'))
        else:
            line.decode('utf-8')

    for line in proc.stderr:
        print(line.decode('utf-8'))

class ShellScript:
    def __init__(self,path="",prefix=""):
        self.path = path
        self.prefix = prefix
        self.clear()
        
    def clear(self):
        self.text = "#!/bin/sh\n"
    
    def add_line(self,cmd):
        self.text += "%s %s\n" % (self.prefix, cmd)
    
    def add_param(self,param):
        l = [self.prefix]
        for k,v in param.items():
            l += [k, str(v)]
        self.text += " ".join(l) + "\n"
        
    def write(self):
        with open(self.path, 'w') as f:
            f.write(self.text)
            
        print("Shell Script generated: " + self.path)
        
        proc = Popen(["chmod","+x",self.path], stdin=None, stdout=PIPE, stderr=PIPE)

        for line in proc.stderr:
            print(line.decode('utf-8'))
            
    def execute(self):
        proc = Popen("./"+self.path, stdin=None)