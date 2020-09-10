import numpy as np
import pandas as pd
import ROOT
from array import array
import sys, os

# setting paths for root, delphes and exroot
ROOT_PATH = os.environ["ROOT_PATH"]
DELPHES_PATH = os.environ["DELPHES_PATH"]
EXROOT_PATH = os.environ["EXROOT_PATH"]

sys.path.append(ROOT_PATH)
sys.path.append(ROOT_PATH + "/bin/")
sys.path.append(ROOT_PATH + "/include/")
sys.path.append(ROOT_PATH + "/lib")

ROOT.gSystem.AddDynamicPath(DELPHES_PATH)
ROOT.gSystem.Load("libDelphes.so");

ROOT.gSystem.AddDynamicPath(EXROOT_PATH)
ROOT.gSystem.Load("libExRootAnalysis.so")

try:
    ROOT.gInterpreter.Declare('#include "' + os.path.join(DELPHES_PATH, 'classes/DelphesClasses.h') + '"')
    ROOT.gInterpreter.Declare('#include "' + os.path.join(EXROOT_PATH, 'ExRootTreeReader.h') + '"')
    print("Delphes classes imported")
except:
    pass


# functions
def DeltaPhi(phi1, phi2):
    """
    Returns the difference of phi1 and phi2
    """
    phi = phi1 - phi2
    if phi >= np.pi:
        phi -= 2 * np.pi
    elif phi < -1 * np.pi:
        phi += 2 * np.pi
    return phi


def MaxValueHist(h1, h2):
    """
    Returns the maximum value from histograms h1 and h2
    """
    return max(h1.GetBinContent(h1.GetMaximumBin()),
               h2.GetBinContent(h2.GetMaximumBin())) * 1.1


def JetMass(jet):
    """
    Returns the invariant mass for a system made out
    of the first two jets in the branch jet
    """
    jets = [ROOT.TLorentzVector(), ROOT.TLorentzVector()]
    [jets[i].SetPtEtaPhiE(jet[i].PT, jet[i].Eta, jet[i].Phi, jet[i].Mass) for i in range(2)]
    return abs((jets[0] + jets[1]).M())


def InvariantMass(jet, i, j):
    """
    Returns the invariant mass for a system made out
    of the i-th and j-th jets in the array jet
    """
    jets = [ROOT.TLorentzVector(), ROOT.TLorentzVector()]
    jets[0].SetPtEtaPhiE(jet[i].PT, jet[i].Eta, jet[i].Phi, jet[i].Mass)
    jets[1].SetPtEtaPhiE(jet[j].PT, jet[j].Eta, jet[j].Phi, jet[j].Mass)
    return abs((jets[0] + jets[1]).M())


def PlotHisto(h, show=True, save=True, savePath="./Histograms/", title=None):
    """
    Plot a singular histogram h
    """
    canvas = ROOT.TCanvas()
    h.SetStats(0)
    if title != None:
        h.SetTitle(title)
    canvas.cd()
    h.Draw()
    if show == True:
        canvas.Draw()
    if save == True:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        name = "{}".format(h.GetName())
        canvas.SaveAs(os.path.join(savePath, name.replace("background", "") + "significance.png"))


def PlotHistos(hs, hb, show=True, save=True, savePath="./Histograms/", title=None):
    """
    Plot histograms hs and hb in the same canvas
    """
    canvas = ROOT.TCanvas()
    canvas.cd()
    hs.Scale(1 / hs.Integral())
    hb.Scale(1 / hb.Integral())
    hs.SetStats(0)
    hb.SetStats(0)
    hb.SetLineColor(2)
    hs.SetFillColor(2)
    hs.SetMaximum(MaxValueHist(hs, hb))
    if title != None:
        hs.SetTitle(title)
    hs.Draw("pe")
    hb.Draw("h,same")

    legend = ROOT.TLegend(0.6, 0.8, 0.9, 0.9)
    legend.AddEntry(hs, "Signal")
    legend.AddEntry(hb, "Background")
    legend.Draw()
    if show == True:
        canvas.Draw()
    if save == True:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        name = "{}".format(hs.GetName())
        canvas.SaveAs(os.path.join(savePath, name.replace("signal", "") + ".png"))


def sigGreaterThan(hz, hs, hb):
    """
    @brief Computes the significance of the "greater than" type
    @param hz Histogram to fill with the significance value
    @param hs Signal histogram
    @param hb Background histogram
    """
    for i in range(1, hz.GetNbinsX()):
        s = hs.Integral(i + 1, hz.GetNbinsX())
        b = hb.Integral(i + 1, hz.GetNbinsX())
        if s + b != 0:
            hz.SetBinContent(i + 1, s / np.sqrt(s + b))
        else:
            hz.SetBinContent(i + 1, 0)
    return hz


def sigLessThan(hz, hs, hb):
    """
    @brief Computes the significance of the "less than" type
    @param hz Histogram to fill with the significance value
    @param hs Signal histogram
    @param hb Background histogram
    """
    for i in range(1, hz.GetNbinsX()):
        s = hs.Integral(1, i + 1)
        b = hb.Integral(1, i + 1)
        if s + b != 0:
            hz.SetBinContent(i + 1, s / np.sqrt(s + b))
        else:
            hz.SetBinContent(i + 1, 0)
    return hz


def Significance(hs, hb, ns=50000, nb=50000, L=25000, sigmab=72.38, sigmas=13.76, title=None, lessThan=False):
    """
    Returns a significance histogram made out
    of the signal and background histograms hs and hb

    L:      integrated luminosity in pb^-1

    ns, nb: Number of signal and background events

    sigmas, sigmab: cross sections for the signal and
                    background events in pb
    """
    hs.Scale((L * sigmas / ns))
    hb.Scale((L * sigmab / nb))

    hz = hb.Clone()
    if title == None:
        hz.SetTitle("{} Significance".format(hs.GetTitle()))
    else:
        hz.SetTitle(title)

    if lessThan == False:
        return sigGreaterThan(hz, hs, hb)
    else:
        return sigLessThan(hz, hs, hb)


def CutFlow(sig_cut, bac_cut, cut_keys, n_cuts=8, ns=50000, nb=50000, L=25000, sigmab=72.38, sigmas=13.76):
    """
    Returns the cut flow tables

    sig_cut,bac_cut:  dictionaries with the signal
                      and background cuts
    cut_keys:         dictionary with the cuts keys
    n_cuts:           number of available cuts
    L:                integrated luminosity in pb^-1

    ns,nb:            number of signal and background events

    sigmas,sigmab:    cross sections for the signal and
                      background events in pb

    """
    # number of applied cuts
    N_cuts = n_cuts - sig_cut.values().count(0)

    # modified cut dics
    s_cut = {key: value for key, value in zip(sig_cut.keys(), sig_cut.values()) if value != 0}
    b_cut = {key: value for key, value in zip(bac_cut.keys(), bac_cut.values()) if value != 0}
    c_keys = {"cut{}".format(i): cut_keys["cut{}".format(i)] for i in range(N_cuts)}

    # weights for signal and background events
    ws = L * sigmas / ns
    wb = L * sigmab / nb

    # dataframes
    s1 = pd.Series(s_cut, dtype=float) if N_cuts != 8 else pd.Series(sig_cut, dtype=float)
    s2 = pd.Series(b_cut, dtype=float) if N_cuts != 8 else pd.Series(bac_cut, dtype=float)
    s3 = pd.Series(c_keys) if N_cuts != 8 else pd.Series(cut_keys)

    # cut flow tables
    df1 = pd.concat([s3, s1, s2], axis=1, keys=["${\\textbf{[bold]: GeV}}$", "S", "B"])

    df1["Z"] = ws * df1["S"] / np.sqrt(ws * df1["S"] + wb * df1["B"])

    df2 = pd.DataFrame(index=["cut{}".format(i) for i in range(N_cuts)],
                       columns=["${\\textbf{[bold]: GeV}}$", "s_c", "s_r", "b_a", "b_r"])

    for i in range(N_cuts):
        df2.iloc[i, 0] = cut_keys["cut{}".format(i)]

    for i in range(1, 5):
        df2.iloc[0, i] = 1

    for i in range(1, N_cuts):
        df2.iloc[i, 1] = df1.iloc[i, 1] / df1.iloc[0, 1]
        df2.iloc[i, 3] = df1.iloc[i, 2] / df1.iloc[0, 2]
        df2.iloc[i, 2] = df1.iloc[i, 1] / df1.iloc[i - 1, 1]
        df2.iloc[i, 4] = df1.iloc[i, 2] / df1.iloc[i - 1, 2]

    return df1, df2

def MaxVal(h1, h2):
    """
    Returns the maximum value from histograms h1 and h2 (h2 is a list of histograms)
    """
    return max(h1.GetBinContent(h1.GetMaximumBin()),
               *[h2[i].GetBinContent(h2[i].GetMaximumBin()) for i in range(len(h2))]) * 1.1

def plotHIstos(hBckg, hSig, plotName, sigLabels, savePath='./'):
    canvas = ROOT.TCanvas()
    canvas.cd()

    hBckg.Scale(1 / hb.Integral())
    hBckg.SetStats(0)
    hBckg.SetLineColor(2)

    for i in range(len(hSig)):
        hSig[i].Scale(1/hs.Integral())
        hSig.SetStats(0)
        hSig.SetFillColor(2)



    hSig[0].SetMaximum(MaxVal(hBckg, hSig))
    hSig[0].Draw("pe")
    for i in range(1, len(hSig)):
        hSig[i].Draw("h,same")
    hBckg.Draw("h,same")

    legend = ROOT.TLegend(0.6,0.8,0.9,0.9)
    legend.AddEntry(hBckg, "Background")
    for i in range(len(hSig)):
        legend.AddEntry(hSig[i], sigLabels[i])
    legend.Draw()

    canvas.SaveAs(os.path.join(savePath, plotName + ".png"))
