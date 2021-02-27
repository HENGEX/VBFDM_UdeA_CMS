# using uproot4 as uproot
# using awkward1 as awkward 

import uproot
import awkward
import numpy as np
import pandas as pd
import uproot_methods

class RootTreeReader:

  def __init__(
    self, 
    path: str, 
    branches: list = None,
    n_values: int = 4, 
    tree_name: str = "Delphes"
    ):

    self.path = path
    self.tree_name = tree_name
    self.n_values = n_values
    self.branches = branches


  @property
  def display_tree(self):
    """Displays the ROOT-tree names"""
    with uproot.open(self.path)[self.tree_name] as tree:
      return tree.show()

  @property
  def num_entries(self) -> int:
    """returns the number of events in the ROOT-tree"""
    with uproot.open(self.path)[self.tree_name] as tree:
      return tree.num_entries 

  @staticmethod
  def set_columns_names(df):
    """
    changes the columns of a DataFrame to be lower case 
    and also replace dots for underscores
    """
    df.columns = df.columns.str.lower().str.replace(".","_")
    return df

  def _get_branch(self, branch) -> pd.DataFrame:
    """read and load a ROOT-tree branch into a pandas DataFrame"""
    with uproot.open(self.path)[self.tree_name] as tree: 
      return tree.arrays(branch, library="pd")

  def data(self) -> pd.DataFrame:
    """returns a pd.DataFrame with branches data"""
    dataframe = pd.DataFrame(index=range(self.num_entries))

    for branch in self.branches:
      df = self._get_branch(branch)
      if df.index.get_level_values("subentry").any():
        df = df.unstack().iloc[:,:self.n_values]
        df.columns = [f"{branch}{i}" for i in range(self.n_values)]
        dataframe = dataframe.join(df)
      else:
        df.reset_index(drop=True, level=1, inplace=True)
        dataframe = dataframe.join(df)
    
    self.dataframe = set_columns_names(dataframe)
    return self.dataframe

  