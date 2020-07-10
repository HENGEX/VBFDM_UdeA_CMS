import re
import uproot
import numpy as np
import pandas as pd


def parseIndex(name):
    match = re.search('\[[0-9]+\]', name)
    return None if match is None else int(match[0].replace('[', '').replace(']', ''))

class Leaf:

    def __init__(self, name):
        index = parseIndex(name)
        self.index = None if index is None else [index]
        self.name = name if self.index is None else re.split('\[',name)[0]

    def addIndex(self, index):
        if self.index is None:
            self.index = [index]
        else:
            self.index.append(index)

class Branch:

    def __init__(self, name):
        self.name = name
        self.leafs = []

    def addLeaf(self, leafName):

        for l in self.leafs:
            if l.name == re.split('\[',leafName)[0]:
                if parseIndex(leafName) is not None:
                    l.addIndex(parseIndex(leafName))
                break
        else:
            self.leafs.append(Leaf(leafName))

    def printLeafs(self):
        for l in self.leafs:

            if l.index is not None:
                s = "{0}.{1}[{2}]".format(self.name, l.name, l.index)
            else:
                s = "{0}.{1}".format(self.name, l.name) 
            
            print(s)


class Data:

    def __init__(self, *leafs):
        self.branches = []
        self.parseBranches(*leafs)
        self.signal = None
        self.dataframe = None

    def setSignal(self, signal_path, tree_name):
        self.signal = uproot.open(signal_path)[tree_name]
        self._setupDataframe()

    
    def _setupDataframe(self):

        self.dataframe = pd.DataFrame(index=range(self.signal.numentries))
        #Create a data frame with a column for each leaf
        for b in self.branches:
            for l in b.leafs:

                name = "{}.{}".format(b.name,l.name)
                #arr = self.signal.array(s) #load a lazy array
                #is a list-like leaf?
                if l.index is not None:
                    print(b.name + "." + l.name + f"[{l.index}]")
                    df = self.signal.arrays(branches=[name], outputtype=pd.DataFrame)
                    self.dataframe = self.dataframe.join(df)
                else:
                    df = self.signal.arrays(branches=[name], outputtype=pd.DataFrame).astype("float64")
                    self.dataframe = self.dataframe.join(df)


                    
        
    
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

# SIGNAL_PATH = "/home/santiago/VBF_DMSimp_spin0_EWKExcluded/Events/run_18/DMSimpSpin0_MY5000_MX1000_07042020.root"
# TREE_NAME = "Delphes"
#
# pt = ("Jet.PT[%d]"%(i) for i in range(2))
# phi = ("Jet.Phi[%d]"%(i) for i in range(2))
# d = Data(*(*pt,*phi,"MissingET.MET","MissingET.Phi"))
# d.setSignal(SIGNAL_PATH, TREE_NAME)
#
# print(d.dataframe.head())
