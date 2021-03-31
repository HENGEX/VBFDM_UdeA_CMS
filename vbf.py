# -*- coding: utf-8 -*-

from root_tree_reader import *
from itertools import combinations

def jet_pairs(tree):
  # zipping jet events 
  jet_events = tree.arrays(["Jet.PT","Jet.Eta","Jet.Phi","Jet.Mass"], how="zip")

  # combinations of jets without replacement
  jet_pairs = awkward.combinations(jet_events.Jet, 2)

  # tuple of pair of jets
  return awkward.unzip(jet_pairs)

def masses(tree):
    """
    invariant mass of every pair of jets 
    """
    j1, j2 = jet_pairs(tree)
    jet_mass = np.sqrt(2*j1.PT*j2.PT*(np.cosh(j1.Eta - j2.Eta) - np.cos(j1.Phi - j2.Phi)))

    # list representation to look for empty values 
    jet_mass = awkward.to_list(jet_mass)
    
    for mass in jet_mass:
        if not mass:
            mass.append(np.nan)
    
    # pandas representation
    jet_mass = awkward.to_pandas(jet_mass).unstack()
    jet_mass.columns = np.arange(len(jet_mass.columns))

    return jet_mass.iloc[:,:6]

def DeltaPhi(phi):
  phi[phi >= np.pi] -= 2*np.pi
  phi[phi < -np.pi] += 2*np.pi

  return np.abs(phi) 


class VBF(RootTreeReader):

  def __init__(self, path, tree_name="Delphes"):
    super().__init__(path, tree_name)
  
  def get_dataframe(self):
    """returns a dataframe with met and jet features"""

    vbf_branches = ["MissingET.MET",
                    "MissingET.Eta",
                    "MissingET.Phi",
                    "Jet.PT",
                    "Jet.Eta",
                    "Jet.Phi",
                    "Jet.Mass",
                    "Jet_size"]

    self.get_branches(vbf_branches)

    self._masses = masses(self.tree)

    self._jet_pt_scalar_sum()
    self._delta_eta_leading_jets()
    self._delta_phi_leading_jets()
    self._max_delta_eta()
    self._max_delta_phi()
    self._delta_phi_met_jet()
    self._invariant_mass()
    self._maximum_invariant_mass()
    self._maximum_invariant_mass_index()
    self._delta_eta_max()
    self._delta_phi_max()

    return self.dataframe

  def _add_column(self, col, name):
    self.dataframe[name] = col

  def _jet_pt_scalar_sum(self):
    """scalar sum of transverse momenta (four jets)"""
    self._add_column(self.dataframe[["jet_pt0","jet_pt1","jet_pt2","jet_pt3"]].sum(axis=1), "H")

  def _delta_eta_leading_jets(self):
    """absolute difference on pseudorapidity between leading jets"""
    self._add_column(np.abs(self.dataframe.jet_eta0 - self.dataframe.jet_eta1), "DeltaEtaJets")

  def _delta_phi_leading_jets(self):
    """absolute difference on azimuthal angle between leading jets"""
    self._add_column(DeltaPhi(self.dataframe.jet_phi0 - self.dataframe.jet_phi1), "DeltaPhiJets")

  def _max_delta_eta(self):
    """maximum absolute difference on pseudorapidity"""
    j1, j2 = jet_pairs(self.tree)
    max_delta = np.abs(j1.Eta - j2.Eta)
    max_delta = awkward.to_pandas(max_delta).unstack()

    self._add_column(max_delta.max(axis=1), "MaxDeltaEtaJets")

  def _max_delta_phi(self):
    """maximum absolute difference on azimuthal angle"""
    j1,j2 = jet_pairs(self.tree)
    max_phi = j1.Phi - j2.Phi
    max_phi = awkward.to_pandas(max_phi).unstack().apply(DeltaPhi)
    
    self._add_column(max_phi.max(axis=1), "MaxDeltaPhiJets")
                  
  def _delta_phi_met_jet(self):
    """minimum absolute difference on azimuthal angle between met and jets"""
    aux_df = pd.DataFrame(index=np.arange(len(self.dataframe)))
    for i in range(4):
      aux_df[f"{i}"] = self.dataframe[f"jet_phi{i}"] - self.dataframe["missinget_phi"]

    self._add_column(DeltaPhi(aux_df.min(axis=1)), "MinDeltaPhiMetJet")

  def _invariant_mass(self):
    """invariant mass for leading jets"""
    self._add_column(self._masses.loc[:,0], "InvMass")

  def _maximum_invariant_mass(self):
    """maximum invariant mass"""
    self._add_column(np.max(self._masses, axis=1), "MaxInvMass")

  def _maximum_invariant_mass_index(self):
    """index of jets with maximum invariant mass"""
    index_map = dict(zip(range(6), [*combinations([0,1, 2, 3], 2)]))

    self.max_index = self._masses.idxmax(axis=1).map(index_map)
    
    self._add_column(self.max_index, "MaxInvIndex")

  def _delta_eta_max(self):
    """absolute difference on pseudorapidity between jets with maximum invariant mass"""
    eta = []
    for row, index in enumerate(self.max_index):
      if not isinstance(index, type(np.nan)):
        eta.append(np.abs(self.dataframe.loc[row, f"jet_eta{index[0]}"] - self.dataframe.loc[row, f"jet_eta{index[1]}"]))
      else:
        eta.append(None)

    self._add_column(eta, "DeltaEtaJetsMax")

  def _delta_phi_max(self):
    """absolute difference on azimuthal angle between jets with maximum invariant mass"""
    phi, nan_phi = {}, {}
    for row, index in enumerate(self.max_index):
      if not isinstance(index, type(np.nan)):
        phi[row] = self.dataframe.loc[row, f"jet_phi{index[0]}"] - self.dataframe.loc[row, f"jet_phi{index[1]}"]
      else:
        nan_phi[row] = np.nan
  
    phi = DeltaPhi(pd.Series(phi))

    if nan_phi:
      nan_phi = pd.Series(nan_phi)
      self._add_column(pd.concat([phi, nan_phi]).sort_index(), "DeltaPhiJetsMax")
    else:
      self._add_column(phi, "DeltaPhiJetsMax")