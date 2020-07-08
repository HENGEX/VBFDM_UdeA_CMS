import re
import uproot
import numpy as np
import pandas as pd


class Leaf:

    def __init__(self, name):        
        self.index = self.parseIndex(name)
        self.name = name if self.index == None else re.split('\[',name)[0]
        
    def parseIndex(self, name):
        match = re.search('\[[0-9]+\]', name)
        return None if match == None else int(match[0].replace('[','').replace(']','')) 
        


class Branch:

    def __init__(self, name):
        self.name = name
        self.leafs = []

    def addLeaf(self, leafName):
        self.leafs.append(Leaf(leafName))

    def printLeafs(self):
        for l in self.leafs:
            
            s = "%s.%s" %(self.name, l.name)            
            print(s)


class Data:

    def __init__(self, *leafs):
        self.branches = []
        self.parseBranches(*leafs)

    def parseBranches(self, *leafs):
        br = self.branches

        if len(leafs) == 0:
            return

        for l in leafs:
            ns = re.split('\.', l)  # split strings with format <label1>.<label2>[...]

            # looking for a previously created branch
            for b in br:
                if b.name == ns[0]:
                    if len(ns) > 1:
                        b.addLeaf(ns[1])
                    break
            else:
                br.append(Branch(ns[0]))
                if len(ns) > 1:
                    br[-1].addLeaf(ns[1])
        return br

    def printData(self):
        for b in self.branches:
            b.printLeafs()

pt = ("Jet.PT[%d]"%(i) for i in range(11))
phi = ("Jet.Phi[%d]"%(i) for i in range(6))
d = Data(*(*pt,*phi,"MET.MET","MET.Phi"))
d.printData()
# d = Data("Jet.PT[0]", "Jet.PT[1]", "MET.MET")
# d.printData()
