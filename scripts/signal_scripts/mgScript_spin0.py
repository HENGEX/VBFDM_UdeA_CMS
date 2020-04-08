 # -*- coding: utf-8 -*-

from os import system
from numpy.random import randint
from numpy import loadtxt

# paths
rcPath = "/home/santiago/VBF_DMSimp_spin0_EWKExcluded/Cards/run_card.dat"
pcPath = "/home/santiago/VBF_DMSimp_spin0_EWKExcluded/Cards/param_card.dat"
mlPath = "./massList.txt"
binPath = "/home/santiago/VBF_DMSimp_spin0_EWKExcluded/bin/generate_events"

# Number of unweighted events 
numOfEvents = 50000

# Initial parameters for the param_card
couplingDic = {
                "gSXr":   "    1 0.000000e+00",
                "gSXc":   "    2 0.000000e+00",
                "gSXd":   "    3 1.000000e+00",
                "gPXd":   "    4 0.000000e+00",
                "gSd11":  "    5 0.250000e+00",
                "gSu11":  "    6 0.250000e+00",
                "gSd22":  "    7 0.250000e+00",
                "gSu22":  "    8 0.250000e+00",
                "gSd33":  "    9 0.250000e+00",
                "gSu33":  "   10 0.250000e+00",
                "gPd11":  "   11 0.000000e+00",
                "gPu11":  "   12 0.000000e+00",
                "gPd22":  "   13 0.000000e+00",
                "gPu22":  "   14 0.000000e+00",
                "gPd33":  "   15 0.000000e+00",
                "gPu33":  "   16 0.000000e+00",
                "Lambda": "   17 1.000000e+04",
                "gSg":    "   18 0.250000e+00",
                "gPg":    "   19 0.000000e+00",
                "gSh1":   "   20 1.000000e+00",
                "gSh2":   "   21 0.000000e+00",
                "gSb":    "   22 1.000000e+00",
                "gPb":    "   23 0.000000e+00",
                "gSw":    "   24 0.250000e+00",
                "gPw":    "   25 0.000000e+00"
            }

def modRunCard():
    """
    Modifica la run_card 
    """
    tempPath = "temp.dat"
    system("cp " + rcPath + " " + tempPath)
    temp = open(tempPath)
    f = open(rcPath,"w")

    for l in temp:
        if "= nevents" in l:
            l = "  {0} ".format(numOfEvents) + l[l.find("= nevents"):]
        elif "= iseed" in l:
            seed = randint(0,65000)
            l = "  {0} ".format(seed)+l[l.find("= iseed"):]
        elif "= use_syst" in l:
            l = "   False " + l[l.find("= use_syst"):]
        f.write(l)

    f.close()
    temp.close()
    system("rm " + tempPath)


def modParamCard(mx, my):
    """
    Modifica la param_card
    """
    system("cp " + pcPath + " temp.dat")
    tempPath = "temp.dat"
    t = open(tempPath, "r")
    f = open(pcPath,"w")  

    for l in t:
        if " # MXd" in l:
            l = "   52 {0}".format(mx) + l[l.find(" # MXd"):]
        elif " # MY0" in l:
            l = "   54 {0}".format(my) + l[l.find(" # MY0"):]
        f.write(l)
    
    f.close()
    t.close()
    system("rm " + tempPath)

def initParamCard():
    """
    Inicializa los parámetros de la param_card
    """
    system("cp " + pcPath + " temp.dat")
    tempPath = "temp.dat"

    t = open(tempPath,"r")
    f = open(pcPath,"w")
    for l in t:
        for d in couplingDic:
            if d in l:
                l = couplingDic[d] + l[l.find(" # " + d):]
        f.write(l)

    f.close()
    t.close()
    system("rm " + tempPath)

initParamCard()
ml = loadtxt(mlPath)

if len(ml) == 2:
    modRunCard()
    modParamCard(ml[0], ml[1])
    system(binPath)
else:
    for i in range(ml.shape[0]):
        modRunCard()
        modParamCard(ml[i,0], ml[i,1])
        system(binPath)
