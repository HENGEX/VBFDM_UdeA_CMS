from root_tree_reader import *
from itertools import combinations


class VBF(RootTreeReader):

  def __init__(self, path, tree_name="Delphes"):
    super().__init__(path, tree_name)
  

  def get_vbf_features(self, n_jets=4):
    """
    returns a dataframe with met and jet features

    n_jets : int
        number of jets requiered
    """
    self.n_jets = n_jets

    vbf_branches = ["MissingET.MET",
                    "MissingET.Eta",
                    "MissingET.Phi",
                    "Jet.PT",
                    "Jet.Eta",
                    "Jet.Phi",
                    "Jet.Mass",
                    "Jet_size"]

    self.dataframe = self.get_branches(vbf_branches, self.n_jets)

    # events with less than two jets
    self._mask = self.dataframe.jet_size < 2

    # jet pairs and invariant masses    
    self.j1, self.j2 = self.jet_pairs()
    self._masses()

    # vbf composite features
    self._jet_pt_scalar_sum()
    self._delta_eta_leading_jets()
    self._delta_phi_leading_jets()
    self._max_delta_eta()
    self._max_delta_phi()
    self._min_delta_phi_met_jet()
    self._invariant_mass()
    self._maximum_invariant_mass()
    self._maximum_invariant_mass_index()
    self._delta_eta_max()    
    self._delta_phi_max()

    return self.dataframe


  def _jet_pt_scalar_sum(self):
    """scalar sum of transverse momenta (four jets)"""
    self._add_column(self.dataframe[["jet_pt0","jet_pt1","jet_pt2","jet_pt3"]].sum(axis=1), "H")


  def _delta_eta_leading_jets(self):
    """absolute difference on pseudorapidity between leading jets"""
    self._add_column(np.abs(self.dataframe.jet_eta0 - self.dataframe.jet_eta1), "DeltaEtaJets")


  def _delta_phi_leading_jets(self):
    """absolute difference on azimuthal angle between leading jets"""
    self._add_column(self.DeltaPhi(self.dataframe.jet_phi0 - self.dataframe.jet_phi1), "DeltaPhiJets")


  def _max_delta_eta(self):
    """maximum absolute difference on pseudorapidity"""
    max_delta = np.abs(self.j1.Eta - self.j2.Eta)
    max_delta = awkward.to_pandas(max_delta).unstack()

    self._add_column(max_delta.max(axis=1), "MaxDeltaEtaJets")


  def _max_delta_phi(self):
    """maximum absolute difference on azimuthal angle"""
    max_phi = self.j1.Phi - self.j2.Phi
    max_phi = awkward.to_pandas(max_phi).unstack().apply(self.DeltaPhi)
    
    self._add_column(max_phi.max(axis=1), "MaxDeltaPhiJets")


  def _min_delta_phi_met_jet(self):
    """minimum absolute difference on azimuthal angle between met and jets"""
    aux = self.DeltaPhi(np.subtract(self.dataframe[["jet_phi0","jet_phi1","jet_phi2","jet_phi3"]], 
                                    self.dataframe[["missinget_phi"]]))

    self._add_column(aux.min(axis=1), "MinDeltaPhiMetJet")


  def _invariant_mass(self):
    """invariant mass for leading jets"""
    self._add_column(self.masses.loc[:,0], "InvMass")


  def _maximum_invariant_mass(self):
    """maximum invariant mass"""
    self._add_column(np.max(self.masses, axis=1), "MaxInvMass")


  def _maximum_invariant_mass_index(self):
    """index of jets with maximum invariant mass"""
    index_map = dict(zip(range(len(self.combinations)), self.combinations))

    self.max_index = self.masses.idxmax(axis=1).map(index_map)
    self._add_column(self.max_index, "MaxInvIndex")


  def _delta_eta_max(self):
    """absolute difference on pseudorapidity between jets with maximum invariant mass"""
    self._max_deltas("eta", "DeltaEtaJetsMax")


  def _delta_phi_max(self):
    """absolute difference on azimuthal angle between jets with maximum invariant mass"""
    self._max_deltas("phi", "DeltaPhiJetsMax", self.DeltaPhi)


  def _add_column(self, col, name):
    """adds a column to self.dataframe"""
    self.dataframe[name] = col 


  def _max_deltas(self, var_name, col_name, func=np.abs):
    """
    joins the absolute difference on var_name for the jets 
    with maximum invariant mass to self.dataframe

    Parameters:
    var_name : str (phi or eta)
        name of the variable
    col_name : str
        name of the column
    func : function (optional)
        function to apply to the variable
    """

    ids = self.dataframe.MaxInvIndex[~self._mask]
    var = pd.DataFrame(index=self.dataframe[~self._mask].index)

    for row, (i,j) in zip(ids.index, ids):
      var.loc[row, "value"] = self.dataframe.loc[row, f"jet_{var_name}{i}"] - self.dataframe.loc[row, f"jet_{var_name}{j}"]

    if not any(self._mask):
      self._add_column(func(var), col_name)

    else:
      nan = pd.DataFrame(index=self.dataframe[self._mask].index)
      nan["value"] = np.nan

      self._add_column(pd.concat([func(var), nan]).sort_index(), col_name)


  def jet_pairs(self):
    """
    returns a tuple of awkward arrays to perform vectorized
    operations with different pairs of jets
    """
    # record array with jagged arrays zipped
    jet_events = self.tree.arrays(["Jet.PT","Jet.Eta","Jet.Phi","Jet.Mass"], how="zip")

    # combinations of jets without replacement
    jet_pairs = awkward.combinations(jet_events.Jet, 2) 

    return awkward.unzip(jet_pairs)


  def _masses(self):
    """
    returns the invariant mass of every pair of n_jets

    tree : TTree
        (uproot) ROOT TTree
    n_jets : int
        number of jets
    """
    jet_masses = self.InvMass(self.j1, self.j2)
    
    jet_masses = awkward.to_pandas(jet_masses).unstack()
    jet_masses.columns = np.arange(len(jet_masses.columns))

    # events with less than two jets
    if any(self._mask):
      empty_masses = pd.DataFrame(
          np.full((len(self.dataframe[self._mask]), len(jet_masses.columns)), np.nan), 
          index=self.dataframe[self._mask].index)
      
      jet_masses = pd.concat([jet_masses, empty_masses]).sort_index()
    
    # combinations of pairs of n_jets
    self.combinations = [*combinations(range(self.n_jets), 2)]

    self.masses = jet_masses.iloc[:, :len(self.combinations)]


  @staticmethod
  def InvMass(j1, j2):
    """di-jet invariant mass"""
    return np.sqrt(2*j1.PT*j2.PT*(np.cosh(j1.Eta - j2.Eta) - np.cos(j1.Phi - j2.Phi)))


  @staticmethod
  def DeltaPhi(dphi):
    """
    correction on azimuthal angle difference dphi
    """
    dphi[dphi >= np.pi] -= 2*np.pi
    dphi[dphi < -np.pi] += 2*np.pi

    return np.abs(dphi)