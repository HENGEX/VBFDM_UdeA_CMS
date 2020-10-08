from uproot_module import Data
import pandas as pd
import numpy as np
import uproot_methods


TREE_NAME = "Delphes"
N_JETS = 4

def DeltaPhi(phi1,phi2):
    """
    Returns the difference of phi1 and phi2
    """
    phi = phi1-phi2
    phi[phi >= np.pi] -= 2*np.pi
    phi[phi < -np.pi] += 2*np.pi
    return phi

def invariant_mass(pt1, pt2, eta1, eta2, phi1, phi2, mass1, mass2):
    pt1 = pt1.to_numpy()
    pt2 = pt2.to_numpy()
    eta1 = eta1.to_numpy()
    eta2 = eta2.to_numpy()
    phi1 = phi1.to_numpy()
    phi2 = phi2.to_numpy()
    mass1 = mass1.to_numpy()
    mass2 = mass2.to_numpy()

    lv1 = uproot_methods.TLorentzVectorArray.from_ptetaphim(pt1, eta1, phi1, mass1)
    lv2 = uproot_methods.TLorentzVectorArray.from_ptetaphim(pt2, eta2, phi2, mass2)
    
    return (lv1 + lv2).mass

#============================
#        Default Cuts
#============================

def cut1(df):
    mask = df["HT"] > 200
    df = df.loc[mask, :]
    return df

def cut2(df):
    # at least two jets per event
    mask = df["Jet.PT[0]"].notnull() & df["Jet.PT[1]"].notnull()
    df = df.loc[mask, :]
    return df

def cut3(df):
    # pt > 30 GeV and |eta| < 5 for leading and subleading jets
    mask = ((df["Jet.PT[0]"] > 30) &
            (df["Jet.PT[1]"] > 30) &
            (abs(df["Jet.Eta[0]"]) < 5) &
            (abs(df["Jet.Eta[1]"]) < 5))
    df = df.loc[mask, :]
    return df

def cut4(df):
    # leading jets in opposite hemispheres
    mask = df["Jet.Eta[0]"] * df["Jet.Eta[1]"] < 0
    df = df.loc[mask, :]
    return df

def cut5(df):
    # |DeltaPhi| >= 2.3 for leading jets
    mask = abs(df["DPhi_J0_J1"]) > 2.3
    df = df.loc[mask, :]
    return df

def cut6(df):
    #     # max invariant mass >= 1000
    mask = df.max_inv_mass >= 1000
    df = df.loc[mask,:]
    return df

def cut7(df):
    mask = np.abs(df[[f"DPhi_MET_J{i}" for i in range(N_JETS)]] ).min(axis=1) >= 0.5
    df = df.loc[mask, :]
    return df

def cut8(df):
    mask = np.abs(df["Jet.Eta[0]"] - df["Jet.Eta[1]"]) < 2.5
    df = df.loc[mask, :]
    return df



class VBFDM:

    def __init__(self, signal_path=None, background_path=None):

        pt = ("Jet.PT[%d]" % (i) for i in range(N_JETS))
        phi = ("Jet.Phi[%d]" % (i) for i in range(N_JETS))
        eta = ("Jet.Eta[%d]" % (i) for i in range(N_JETS))
        mass = ("Jet.Mass[%d]" % (i) for i in range(N_JETS))
        leafs = (*pt,*phi,*eta,*mass,"MissingET.MET","MissingET.Phi","MissingET.Eta")
        self.signal = Data(*leafs)
        self.background = Data(*leafs)

        # Setting the signal and the background
        if signal_path is not None:
            self.add_signal(signal_path)
        if background_path is not None:
            self.add_background(background_path)

        # Add new columns
        ht = lambda df: df[["Jet.PT[{}]".format(i) for i in range(N_JETS)]].sum(axis=1)
        self.add_column("HT", ht)

        fc = lambda df: DeltaPhi(df["Jet.Phi[0]"], df["Jet.Phi[1]"])
        self.add_column("DPhi_J0_J1", fc)

        for i in range(N_JETS):
            fc = lambda df: DeltaPhi(df[f"Jet.Phi[{i}]"], df["MissingET.Phi"])
            self.add_column(f"DPhi_MET_J{i}", fc)

        # Invariant mass
        n = []
        for i in range(N_JETS):
            for j in range(i+1,N_JETS):
                fc = lambda df: invariant_mass(df[f"Jet.PT[{i}]"],
                                               df[f"Jet.PT[{j}]"],
                                               df[f"Jet.Eta[{i}]"],
                                               df[f"Jet.Eta[{j}]"],
                                               df[f"Jet.Phi[{i}]"],
                                               df[f"Jet.Phi[{j}]"],
                                               df[f"Jet.Mass[{i}]"],
                                               df[f"Jet.Mass[{j}]"])
                n.append(f"InvMass_J{i}_J{j}")
                self.add_column(f"InvMass_J{i}_J{j}",fc)

        fc = lambda df: df[n].max(axis=1)
        self.add_column("max_inv_mass",fc)

        

    def add_signal(self, path):
        self.signal.setSignal(path, TREE_NAME)

    def add_background(self, path):
        self.background.setSignal(path, TREE_NAME)

    def add_column(self, col_name, col_func):
        """
        Add a new column of data to background and signal
        data frames
        :param col_name: Name of the new column
        :param col_func: Function to compute the values of the column. This
                         function must receives a data frame as input and
                         returns a serie
        """
        self.signal.dataframe[col_name] = col_func(self.signal.dataframe)
        self.background.dataframe[col_name] = col_func(self.background.dataframe)

    def add_cut(self, cut_name, cut_func):
        self.signal.addCut(cut_name, cut_func)
        self.background.addCut(cut_name, cut_func)

    def cut_flow(self):
        cuts = { r'$H_T > 200$': cut1,
                 r'$N^o Jets \geq 2$': cut2,
                 r'$P_T(J_i)>30\ \ and\ \eta(J_i)<5\ i=0,1$': cut3,
                 r'$\eta (J_0)*\eta (J_1) < 0$': cut4,
                 r'$|\Delta \phi (J_0,J_1)| > 2.3$': cut5,
                 r'$max(m(J_i,J_j)) > 1000$': cut6,
                 r'$min(|\Delta\phi(MET,J_i)|) > 0.5,\ i=0...4$': cut7,
                 r'$|\Delta \eta (J_0,J_1)| < 2.5$': cut8}

        for c in cuts:
            self.add_cut(c,cuts[c])

        dsignal = self.signal.cutFlow()
        dbackground = self.background.cutFlow()

        s1 = pd.Series(dsignal, name="sig", dtype=float)
        s2 = pd.Series(dbackground, name="back", dtype=float)
        df = pd.concat([s1, s2], axis=1)
        return df

