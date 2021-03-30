# using uproot4 as uproot
# using awkward1 as awkward 

import uproot
import awkward
import pandas as pd


class RootTreeReader:

  """ 
  Read data from a ROOT tree

  Parameters:
  -----------
  path : string
    Path to the ROOT tree

  tree_name : string (default=Delphes)
    Name of the ROOT tree

  Attributes:
    tree: Root tree read
  """

  def __init__(self, path: str, tree_name: str = "Delphes"):
    self.tree = uproot.open(path)[tree_name]


  def get_branch(self, branch: str, library="pd"):
    """
    returns a tree branch element 

    branch: string
      Branch to load from the ROOT tree

    library: string (default="pd")
      The library that is used to represent the data
      pd : pandas dataframe
      ak : awkward array
      np : numpy array
    """
    return self.tree.arrays(branch, library=library)


  def get_dataframe(self, branches: list, n_values: int = 4) -> pd.DataFrame:
    """
    returns a pandas dataFrame with branches as columns

    branches : array-like
      Branches to load from the ROOT tree

    n_values : int (default=4)
      Number of values to load from jagged arrays
    """
    self.branches = branches
    self.n_values = n_values
    self.dataframe = pd.DataFrame(index=range(self.tree.num_entries))

    for branch in self.branches:
      if type(self.tree[branch]) is uproot.models.TBranch.Model_TBranch_v13:
        self._add_branch(branch)
      else:
        self._join_dataframe(branch)
    
    return self._set_columns_names(self.dataframe)


  def _join_dataframe(self, branch):
    """joins branch element to self.dataframe"""
    df = self.get_branch(branch)

    if df.index.get_level_values("subentry").any():
      
      df = df.unstack().iloc[:,:self.n_values]
      df.columns = [f"{branch}{i}" for i in range(self.n_values)]
      self.dataframe = self.dataframe.join(df)

    else:
      df.reset_index(drop=True, level=1, inplace=True)
      self.dataframe = self.dataframe.join(df)


  def _add_branch(self, branch: str, name=None):
    """adds a branch to self.dataframe"""
    if not name:
      name = branch.lower()

    self.dataframe[name] = self.tree[branch].array(library="pd")


  @staticmethod
  def _set_columns_names(df):
    """
    changes the column names of self.dataFrame
    """
    df.columns = df.columns.str.lower().str.replace(".","_")
    return df