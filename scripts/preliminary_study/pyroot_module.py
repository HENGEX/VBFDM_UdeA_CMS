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
sys.path.append(ROOT_PATH+"/bin/")
sys.path.append(ROOT_PATH+"/include/")
sys.path.append(ROOT_PATH+"/lib")

ROOT.gSystem.AddDynamicPath(DELPHES_PATH)
ROOT.gSystem.Load("libDelphes.so");

ROOT.gSystem.AddDynamicPath(EXROOT_PATH)
ROOT.gSystem.Load("libExRootAnalysis.so")

try:
    ROOT.gInterpreter.Declare('#include "'+os.path.join(DELPHES_PATH, 'classes/DelphesClasses.h')+'"')
    ROOT.gInterpreter.Declare('#include "'+os.path.join(EXROOT_PATH, 'ExRootTreeReader.h')+'"')
    print("Delphes classes imported")
except:
    pass


# functions
def DeltaPhi(phi1,phi2):
    """
    Returns the difference of phi1 and phi2
    """
    phi = phi1-phi2
    if phi >= np.pi:
        phi -= 2*np.pi
    elif phi < -1*np.pi:
        phi += 2*np.pi
    return phi

def MaxValueHist(h1, h2):
    """
    Returns the maximum value from histograms h1 and h2
    """
    return max(h1.GetBinContent(h1.GetMaximumBin()),
               h2.GetBinContent(h2.GetMaximumBin()))*1.1

def JetMass(jet):
    """
    Returns the invariant mass for a system made out 
    of the first two jets in the branch jet
    """
    jets = [ROOT.TLorentzVector(), ROOT.TLorentzVector()]
    [jets[i].SetPtEtaPhiE(jet[i].PT, jet[i].Eta, jet[i].Phi, jet[i].Mass) for i in range(2)]
    return abs((jets[0]+jets[1]).M())      

def InvariantMass(jet,i,j):
    """
    Returns the invariant mass for a system made out 
    of the i-th and j-th jets in the array jet
    """
    jets = [ROOT.TLorentzVector(), ROOT.TLorentzVector()]
    jets[0].SetPtEtaPhiE(jet[i].PT, jet[i].Eta, jet[i].Phi, jet[i].Mass)
    jets[1].SetPtEtaPhiE(jet[j].PT, jet[j].Eta, jet[j].Phi, jet[j].Mass)
    return abs((jets[0]+jets[1]).M())
    
def PlotHisto(h, show=True, save=True, savePath="./Histograms/", title=None):
    """
    Plot a singular histogram h
    """
    canvas = ROOT.TCanvas()
    h.SetStats(0)
    if title!=None:
        h.SetTitle(title)
    canvas.cd()
    h.Draw()
    if show==True:
        canvas.Draw()
    if save==True:
        if savePath == "./Histograms/":
            os.system("mkdir Histograms")
        name = "{}".format(h.GetName())
        canvas.SaveAs(savePath+name.replace("background","")+"significance.png")

def PlotHistos(hs, hb, show=True, save=True, savePath="./Histograms/", title=None):
    """
    Plot histograms hs and hb in the same canvas
    """
    canvas = ROOT.TCanvas()
    canvas.cd()
    hs.Scale(1/hs.Integral())
    hb.Scale(1/hb.Integral())
    hs.SetStats(0)
    hb.SetStats(0)
    hb.SetLineColor(2)
    hs.SetFillColor(2)
    hs.SetMaximum(MaxValueHist(hs, hb))
    if title!=None:
        hs.SetTitle(title)
    hs.Draw("pe")
    hb.Draw("h,same")

    legend = ROOT.TLegend(0.6,0.6,0.9,0.9)
    legend.AddEntry(hs, "Signal")
    legend.AddEntry(hb, "Background")
    legend.Draw()
    if show==True:
        canvas.Draw()
    if save==True:
        if savePath == "./Histograms/":
            os.system("mkdir Histograms")
        name = "{}".format(hs.GetName())
        canvas.SaveAs(savePath+name.replace("signal","")+".png")

def Significance(hs, hb, ns=50000, nb=50000, L=25000, sigmab=72.38, sigmas=13.76, title=None):
    """
    Returns a significance histogram made out 
    of the signal and background histograms hs and hb
    
    L:      integrated luminosity in pb^-1
    
    ns, nb: Number of signal and background events
    
    sigmas, sigmab: cross sections for the signal and
                    background events in pb
    """
    hs.Scale((L*sigmas/ns))
    hb.Scale((L*sigmab/nb))
    
    hz = hb.Clone()
    if title==None:
        hz.SetTitle("{} Significance".format(hs.GetTitle()))
    else:
        hz.SetTitle(title)
        
    for i in range(1,hz.GetNbinsX()):
        s = hs.Integral(i+1,hz.GetNbinsX())
        b = hb.Integral(i+1,hz.GetNbinsX())
        if s+b != 0:
            hz.SetBinContent(i+1,s/np.sqrt(s+b))
        else:
            hz.SetBinContent(i+1,0)

    return hz

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
    s_cut = {key:value for key, value in zip(sig_cut.keys(), sig_cut.values()) if value !=0}
    b_cut = {key:value for key, value in zip(bac_cut.keys(), bac_cut.values()) if value !=0}
    c_keys = {"cut{}".format(i):cut_keys["cut{}".format(i)] for i in range(N_cuts)}

    # weights for signal and background events
    ws = L*sigmas/ns
    wb = L*sigmab/nb

    # dataframes
    s1 = pd.Series(s_cut, dtype=float) if N_cuts!=8 else pd.Series(sig_cut, dtype=float)
    s2 = pd.Series(b_cut, dtype=float) if N_cuts!=8 else pd.Series(bac_cut, dtype=float)
    s3 = pd.Series(c_keys) if N_cuts!=8 else pd.Series(cut_keys) 
    
    # cut flow tables
    df1 = pd.concat([s3,s1,s2], axis=1, keys=["${\\textbf{[bold]: GeV}}$","S","B"])
    
    df1["Z"] = ws*df1["S"]/np.sqrt(ws*df1["S"]+wb*df1["B"]) 

    df2 = pd.DataFrame(index=["cut{}".format(i) for i in range(N_cuts)],
                       columns=["${\\textbf{[bold]: GeV}}$","s_c","s_r","b_a","b_r"])
    
    for i in range(N_cuts):
        df2.iloc[i,0] = cut_keys["cut{}".format(i)] 
        
    for i in range(1,5):
        df2.iloc[0,i] = 1

    for i in range(1,N_cuts):
        df2.iloc[i,1] = df1.iloc[i,1]/df1.iloc[0,1]
        df2.iloc[i,3] = df1.iloc[i,2]/df1.iloc[0,2]
        df2.iloc[i,2] = df1.iloc[i,1]/df1.iloc[i-1,1]
        df2.iloc[i,4] = df1.iloc[i,2]/df1.iloc[i-1,2]

    return df1, df2

# classes to build jet and met objects 
class GetJetObject:

    def __init__(self, jet):
        
        self.PT = jet.Pt()
        self.Eta = jet.Eta()
        self.Phi = jet.Phi()
        self.Mass = jet.M()
            
class GetMetObject:

    def __init__(self, met):
 
        self.MET = met[0]
        self.Eta = met[1]
        self.Phi = met[2]

        
# class to process signal and background data
class Data:
        
    # default value for non-default cuts
    apply_cut = False

    # number of jets to book histograms
    numJets = 6

    # histograms bins, minimum and maximum limits
    # to use in the method Histograms 
    hPT_feature = [100,0.0,1800]
    hEta_feature = [50,-5,5]
    hDeltaEta_feature = [50,0,10]
    hDeltaPhi_feature = [50,-1,4]
    hDeltaPhiMetJet_feature = [50,0,5]
    hMass_feature = [25,0,4000]
    hPTdiv_feature = [50,0,20]
    hMetET_feature = [100,0,2000]
    hMetPhi_feature = [50,-5,5]
    hDeltaEtaJet_feature = [25,0,10]
    hDeltaPhiMax_feature = [50,-1,4]
    hDeltaEtaMax_feature = [50,0,10]
    hMassMax_feature = [25,0,4000]
    hPTdivMax_feature = [200,0,4]
    hDeltaEtaJetMax_feature = [25,0,10]    
    
    def __init__(self, data_path, name, tree_name="Delphes"):
        
        self.name = name
        self.treeName = tree_name
	self.chain = ROOT.TChain(tree_name)
	self.chain.Add(data_path)
	self.number_of_events = self.chain.GetEntries()
        
    def DisplayHistograms(self):
        """
        Display histograms description
        """
        try:
            self.hist_names = np.loadtxt("histograms.txt", dtype=str, delimiter=",")
            for name in self.hist_names:
                print(name)
        except:
            pass 

    def Histograms(self):
        """
        Book histograms for VBF objects/variables
        """
	self.hPT = [ROOT.TH1F("PT[{}]_{}".format(i+1,self.name+self.treeName),
                              "jet[{}] P_T".format(i+1),
                              self.hPT_feature[0],
                              self.hPT_feature[1],
                              self.hPT_feature[2])
                    for i in range(self.numJets)]
        
	self.hEta = [ROOT.TH1F("Eta[{}]_{}".format(i+1,self.name+self.treeName),
                               "jet[{}] \eta".format(i+1),
                               self.hEta_feature[0],
                               self.hEta_feature[1],
                               self.hEta_feature[2])
                     for i in range(self.numJets)]
        
	self.hDeltaEta = [ROOT.TH1F("DeltaEtaMetJet[{}]_{}".format(i+1,self.name+self.treeName),
                                    "|\Delta\eta(j{},MET)|".format(i+1),
                                    self.hDeltaEta_feature[0],
                                    self.hDeltaEta_feature[1],
                                    self.hDeltaEta_feature[2])
                          for i in range(self.numJets)]
        
        self.hDeltaPhiMetJet = [ROOT.TH1F("DeltaPhiMetJet[{}]_{}".format(i+1,self.name+self.treeName),
                                          "|\Delta\phi(j{},MET)|".format(i+1),
                                          self.hDeltaPhiMetJet_feature[0],
                                          self.hDeltaPhiMetJet_feature[1],
                                          self.hDeltaPhiMetJet_feature[2])
                                for i in range(self.numJets)]
        
        self.hDeltaPhi = ROOT.TH1F("DeltaPhi_{}".format(self.name+self.treeName),
                                   "|\Delta\phi(jet1,jet2)|",
                                   self.hDeltaPhi_feature[0],
                                   self.hDeltaPhi_feature[1],
                                   self.hDeltaPhi_feature[2])
        
	self.hMass = ROOT.TH1F("Mass_{}".format(self.name+self.treeName),
                               "M(j1,j2)",
                               self.hMass_feature[0],
                               self.hMass_feature[1],
                               self.hMass_feature[2])
        
	self.hPTdiv = ROOT.TH1F("PTdiv_{}".format(self.name+self.treeName),
                                "P_{T}(j1)/P_{T}(j2)", 
                                self.hPTdiv_feature[0],
                                self.hPTdiv_feature[1],
                                self.hPTdiv_feature[2])
        
	self.hMetET = ROOT.TH1F("MetET_{}".format(self.name+self.treeName),
                                "MET_ET", 
                                self.hMetET_feature[0],
                                self.hMetET_feature[1],
                                self.hMetET_feature[2])
        
	self.hMetPhi = ROOT.TH1F("MetPhi_{}".format(self.name+self.treeName),
                                 "MET_PHI", 
                                 self.hMetPhi_feature[0],
                                 self.hMetPhi_feature[1],
                                 self.hMetPhi_feature[2])
        
        self.hDeltaEtaJet = ROOT.TH1F("DeltaEtaJet_{}".format(self.name+self.treeName),
                                      "|\Delta\eta(j1,j2)|",
                                      self.hDeltaEtaJet_feature[0],
                                      self.hDeltaEtaJet_feature[1],
                                      self.hDeltaEtaJet_feature[2])
        
        self.hDeltaPhiMax = ROOT.TH1F("DeltaPhiMax_{}".format(self.name+self.treeName),
                                      "|\Delta\phi(j_i,j_j)| Max", 
                                      self.hDeltaPhiMax_feature[0],
                                      self.hDeltaPhiMax_feature[1],
                                      self.hDeltaPhiMax_feature[2])
        
        self.hMassMax = ROOT.TH1F("MassMax_{}".format(self.name+self.treeName),
                                  "M(ji,jj) Max", 
                                  self.hMassMax_feature[0],
                                  self.hMassMax_feature[1],
                                  self.hMassMax_feature[2])
        
        self.hPTdivMax = ROOT.TH1F("PTdivMax_{}".format(self.name+self.treeName),
                                   "P_{T}(ji)/P_{T}(jj) Max", 
                                   self.hPTdivMax_feature[0],
                                   self.hPTdivMax_feature[1],
                                   self.hPTdivMax_feature[2])
        
        self.hDeltaEtaJetMax = ROOT.TH1F("hDeltaEtaJetMax_{}".format(self.name+self.treeName),
                                         "|\Delta\eta(jj,MET)| Max", 
                                         self.hDeltaEtaJetMax_feature[0],
                                         self.hDeltaEtaJetMax_feature[1],
                                         self.hDeltaEtaJetMax_feature[2])


    def Convert(self, outFileName, treeName="vbf_udea_tree"):

        # Reading  tree branches
	tree = ROOT.ExRootTreeReader(self.chain)
	jet = tree.UseBranch("Jet")
	met = tree.UseBranch("MissingET")
        
        # defining a new tree and the file to store the data
        newTree = ROOT.TTree(treeName, "newTree")
        myFile = ROOT.TFile(outFileName, "RECREATE")

        # variables to store jets and met
        Jets = [ROOT.TLorentzVector()
                for _ in range(self.n_jets)]

        MetMET = array('f', [0])
        MetPhi = array('f', [0])
        MetEta = array('f', [0])

        # setting new tree branches
        [newTree.Branch('Jet{}'.format(i), Jets[i])
         for i in range(self.n_jets)]
                
        newTree.Branch('MetPhi', MetPhi, 'MetPhi/F')
        newTree.Branch('MetEta', MetEta, 'MetEta/F')
        newTree.Branch('MetMET', MetMET, 'MetMET/F')

        none = [-1, -1, -1, -1]
        
        # looping over events 
        for event in range(self.number_of_events):
            # load event
            tree.ReadEntry(event)
            
            N_jets = jet.GetEntries()

            # writing data to a new tree
            if N_jets == 0:
                [Jets[i].SetPtEtaPhiM(*none) for i in range(self.n_jets)]

            if N_jets == self.n_jets:
                [Jets[i].SetPtEtaPhiM(jet.At(i).PT,jet.At(i).Eta,jet.At(i).Phi,jet.At(i).Mass) for i in range(self.n_jets)]    

            for n in range(1,self.n_jets):
                if N_jets == n:
                    [Jets[i].SetPtEtaPhiM(jet.At(i).PT,jet.At(i).Eta,jet.At(i).Phi,jet.At(i).Mass) if i<n else Jets[i].SetPtEtaPhiM(*none) for i in range(self.n_jets)]
                
            MetPhi[0] = met.At(0).Phi
            MetEta[0] = met.At(0).Eta
            MetMET[0] = met.At(0).MET

            # filling new tree
            newTree.Fill()

        #Writing and closing the files
        newTree.Write()
        myFile.Close()
        
    # build jet and met objects
    def GetObjects(self):

        Jets, MET = {}, {}
        J, M = {}, {}

        if self.treeName == "Delphes":

            # Reading  tree branches
	    self.tree = ROOT.ExRootTreeReader(self.chain)
	    self.jet = self.tree.UseBranch("Jet")
	    self.met = self.tree.UseBranch("MissingET")

            for event in range(self.number_of_events):
                
                self.tree.ReadEntry(event)
                
                Jets[event] = [self.jet.At(i) for i in range(self.jet.GetEntries())]
                MET[event] = self.met.At(0)
                
            return Jets, MET
        
        else:
            
            for event in range(self.number_of_events):
            
                self.chain.GetEntry(event)
            
                # reading a vbf_udea_tree
                jet = [getattr(self.chain, "Jet{}".format(i))
                       for i in range(self.numJets)]
            
                MET[event] = [getattr(self.chain, "Met{}".format(i))
                              for i in ["MET","Eta","Phi"]]

                Jets[event] = [jets for jets in jet if jets.M() > 0]

                J[event] = [GetJetObject(jet) for jet in Jets[event]]
                
                M[event] = GetMetObject(MET[event])
            
            return J, M
        
    def Fill(self, jets, met, jet_cut=True, deltaEta_cut=None, invmass_cut=None, met_cut=None, h_cut=None, delta_phi_cut=None,
             jet_cut_value=2, deltaEta_cut_value=4, invmass_cut_value=1000, met_cut_value=200, h_cut_value=200, delta_phi_cut_value=0.5):        


        # setting initial cuts values to class variable apply_cut
        #cut_list = [deltaEta_cut, invmass_cut, met_cut, h_cut, delta_phi_cut]
        #for cut in cut_list:
        #    if cut == None:
        #        cut = self.apply_cut

        if deltaEta_cut == None:
            deltaEta_cut = self.apply_cut
        if invmass_cut == None:
            invmass_cut = self.apply_cut
        if met_cut == None:
            met_cut = self.apply_cut
        if h_cut == None:
            h_cut = self.apply_cut
        if delta_phi_cut == None:
            delta_phi_cut = self.apply_cut
            
        # creating dictionaries for the cuts
        self.math_cuts = ["Number of jets $\geq$ {}".format(jet_cut_value),
                          "$\eta (j_1) * \eta (j_2) < 0$",
                          "$|\Delta \eta (j_1,j_2)| > {}$".format(deltaEta_cut_value),
                          "$\mathbf{{M(j_1,j_2) >}}{}$".format(invmass_cut_value),
                          "$\mathbf{{MET >}}{}$".format(met_cut_value),
                          "$\mathbf{{H_t >}}{}$".format(h_cut_value),
                          "|$\Delta\phi(\text{{MET}},j)| > {}$".format(delta_phi_cut_value)]

        self.cuts_keys = {"cut{}".format(i):key for i,key in zip(range(1,8), self.math_cuts)}
        self.cuts_keys["cut0"] = "${\mathbf{P_T>30}}$, $|\eta(j)|<5$"
        self.cuts =  {"cut{}".format(i):0 for i in range(1,8)}
        self.cuts["cut0"] = self.number_of_events

        self.Jets = {}
        self.MET = {}
            
        # looping over events 
        for event in range(self.number_of_events):
            # load event
            if self.treeName == "Delphes":
                self.tree.ReadEntry(event)
            else:
                self.chain.GetEntry(event)                
                
            # cut0:  jets with tranverse momentum greater
            # than 30 GeV and absolute pseudorapidity smaller than 5
            self.Jets[event] = [jet for jet in jets[event] if jet.PT > 30 and abs(jet.Eta) < 5]
            self.MET[event] = met[event]

            if jet_cut == True:
                # Cut1: minimum number of jets per event
                if len(self.Jets[event]) < jet_cut_value: continue
                self.cuts["cut1"] += 1
            
                # Cut2: jets in opposite hemispheres
    	        if (self.Jets[event][0].Eta * self.Jets[event][1].Eta) >= 0:
                    continue
                self.cuts["cut2"] += 1
        
            # Cut3: difference on pseudorapidity for the leading jets
            if deltaEta_cut == True:
                if abs(self.Jets[event][0].Eta-self.Jets[event][1].Eta) < deltaEta_cut_value:
                    continue
                self.cuts["cut3"] += 1

            # Cut4: invariant mass for the leading jets  
            if invmass_cut == True:
                if InvariantMass(self.Jets[event],0,1) < invmass_cut_value:
                    continue
                self.cuts["cut4"] += 1

            # Cut5: missing energy
            if met_cut == True:
                if self.MET[event].MET < met_cut_value:continue
                self.cuts["cut5"] += 1

                                   
            NJets = len(self.Jets[event]) if len(self.Jets[event]) <= self.numJets else self.numJets
                
            # cut6: 
            if h_cut == True:
                h = 0
                for jet in range(NJets):
                    h += abs(self.Jets[event][jet].PT)
                if h < h_cut_value: continue
                self.cuts["cut6"] += 1

            # cut7: difference on azimuthal angle
            # between MET and jets
            if delta_phi_cut == True:
                delta = []
                for jet in range(NJets):
                    d = abs(DeltaPhi(self.MET[event].Phi,self.Jets[event][jet].Phi))
                    if d < 0.5:
                        delta.append(False)
                    else:
                        delta.append(True)
                if np.sum(delta) < 4: continue
                self.cuts["cut7"] += 1
            
            i, j, JsMass = 0, 0, 0
      	    for n in range(NJets):
       	        self.hPT[n].Fill(self.Jets[event][n].PT)
       	        self.hDeltaPhiMetJet[n].Fill(abs(DeltaPhi(self.MET[event].Phi,self.Jets[event][n].Phi)))
       	        self.hDeltaEta[n].Fill(abs(self.MET[event].Eta-self.Jets[event][n].Eta))

                # looking for the jets that forms the system
                # with the greatest invariant mass 
                for m in range(len(self.Jets[event])):
                    if n==m: continue
                    a = InvariantMass(self.Jets[event],n,m)
                    if a > JsMass:
                        i,j = m,n
                        JsMass = a
                        
  	    self.hPTdiv.Fill(self.Jets[event][0].PT/self.Jets[event][1].PT)
            self.hMetET.Fill(self.MET[event].MET)
  	    self.hMetPhi.Fill(self.MET[event].Phi)
  	    self.hMass.Fill(JetMass(self.Jets[event]))
            self.hDeltaPhi.Fill(abs(DeltaPhi(self.Jets[event][0].Phi,self.Jets[event][1].Phi)))
            self.hDeltaEtaJet.Fill(abs(self.Jets[event][0].Eta-self.Jets[event][1].Eta))
            
            self.hPTdivMax.Fill(self.Jets[event][i].PT/self.Jets[event][j].PT)
            self.hMassMax.Fill(JsMass)
            self.hDeltaPhiMax.Fill(abs(DeltaPhi(self.Jets[event][i].Phi,self.Jets[event][j].Phi)))
            self.hDeltaEtaJetMax.Fill(abs(self.Jets[event][i].Eta-self.Jets[event][j].Eta))   
