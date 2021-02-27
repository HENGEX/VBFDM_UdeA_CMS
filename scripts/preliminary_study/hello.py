import uproot
import awkward
import numpy as np
import pandas as pd
import uproot_methods

class ROOTTreeReader:

  # number of values to read from a jagged branch
  N_VALUES = 4

  def __init__(self, path: str, branches: list = None, tree_name: str = "Delphes"):
    self.path = path
    self.tree_name = tree_name
    self.branches = branches

  def _read_tree_branch(self, branch):
    """reads the ROOT tree branches"""
    with uproot.open(self.path)[self.tree_name] as tree: 
      return tree.arrays(branch, library="pd")

  @property
  def display_tree(self):
    """Displays the tree names"""
    with uproot.open(self.path)[self.tree_name] as tree:
      return tree.show()

  @property
  def num_entries(self):
    """Number of events in the ROOT tree"""
    with uproot.open(self.path)[self.tree_name] as tree:
      return tree.num_entries 

  @property
  def data(self):
    """returns a pd.DataFrame with branches data"""
    dataframe = pd.DataFrame(index=range(self.num_entries))

    for branch in self.branches:
      df = self._read_tree_branch(branch)
      if df.index.get_level_values("subentry").any():
        df = df.unstack().iloc[:,:self.N_VALUES]
        df.columns = [f"{branch}{i}" for i in range(self.N_VALUES)]
        dataframe = dataframe.join(df)
      else:
        df.reset_index(drop=True, level=1, inplace=True)
        dataframe = dataframe.join(df)
    
    return set_columns_names(dataframe)