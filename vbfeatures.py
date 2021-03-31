from itertools import combinations

def jet_pairs(tree):
    """
    returns a tuple to form every combination of jets

    Parameters:
    tree: ROOT TTree
    """
    # zipping jet events 
    jet_events = tree.arrays(["Jet.PT","Jet.Eta","Jet.Phi","Jet.Mass"], how="zip")

    # combinations of two jets without replacement
    jet_pairs = awkward.combinations(jet_events.Jet, 2)

    return awkward.unzip(jet_pairs)

def masses(tree):
    """
    returns the invariant mass of every pair of the four leading jets

    Parameters:
    tree: ROOT TTree 
    """
    j1, j2 = jet_pairs(tree)
    jet_mass = np.sqrt(2*j1.PT*j2.PT*(np.cosh(j1.Eta - j2.Eta) - np.cos(j1.Phi - j2.Phi)))

    masses = awkward.to_pandas(jet_mass).unstack()
    masses.columns = np.arange(len(masses.columns))

    return masses.iloc[:,:6]

def DeltaPhi(phi):
    """returns a correction on the azimuthal angle"""
    phi[phi >= np.pi] -= 2*np.pi
    phi[phi < -np.pi] += 2*np.pi

    return np.abs(phi) 


class VBFeatures(RootTreeReader):

  BRANCHES = ["MissingET.MET", 
              "MissingET.Eta",
              "MissingET.Phi",
              "Jet.PT",
              "Jet.Eta",
              "Jet.Phi",
              "Jet.Mass",
              "Jet_size"]

  def __init__(self, path, tree_name="Delphes"):
    super().__init__(path, tree_name)

  
  def get_dataframe(self):
    """
    returns a dataframe with met and jet features
    """
    self.get_branches(self.BRANCHES)
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

    return self._dataframe

  def _add_column(self, col, name):
    """adds a column to self._dataframe"""
    self._dataframe[name] = col

  def _jet_pt_scalar_sum(self):
    """scalar sum of transverse momenta (four jets)"""
    self._add_column(self._dataframe[["jet_pt0","jet_pt1","jet_pt2","jet_pt3"]].sum(axis=1), "H")

  def _delta_eta_leading_jets(self):
    """absolute difference on pseudorapidity between leading jets"""
    self._add_column(np.abs(self._dataframe.jet_eta0 - self._dataframe.jet_eta1), "DeltaEtaJets")

  def _delta_phi_leading_jets(self):
    """absolute difference on azimuthal angle between leading jets"""
    self._add_column(DeltaPhi(self._dataframe.jet_phi0 - self._dataframe.jet_phi1), "DeltaPhiJets")

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
    aux_df = pd.DataFrame(index=np.arange(len(self._dataframe)))
    for i in range(4):
      aux_df[f"{i}"] = self._dataframe[f"jet_phi{i}"] - self._dataframe["missinget_phi"]

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
    for row, (i,j) in enumerate(self.max_index):
      eta.append(np.abs(self._dataframe.loc[row, f"jet_eta{i}"] - self._dataframe.loc[row, f"jet_eta{j}"]))

    self._add_column(eta, "DeltaEtaJetsMax")

  def _delta_phi_max(self):
    """absolute difference on azimuthal angle between jets with maximum invariant mass"""
    phi = []
    for row, (i,j) in enumerate(self.max_index):
      phi.append(self._dataframe.loc[row, f"jet_phi{i}"] - self._dataframe.loc[row, f"jet_phi{j}"])
      
    self._add_column(DeltaPhi(np.array(phi)), "DeltaPhiJetsMax")