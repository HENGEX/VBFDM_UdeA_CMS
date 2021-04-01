# using uproot4 as uproot
# using awkward1 as awkward 

import numpy as np
import pandas as pd
import uproot
import awkward


class RootTreeReader:

  """ 
  Read data from a ROOT TTree 

  Parameters:
  path : string
      Path to the ROOT file

  tree_name : string (default=Delphes)
      Name of the ROOT TTree

  Attributes:
    tree: Root TTree 
  """

  def __init__(self, path: str, tree_name: str = "Delphes"):
    self.tree = uproot.open(path)[tree_name]


  def get_branches(self, branches: list, max_elements=4):
    """
    returns a DataFrame with branches as features

    branches : array-like
      branches to load from the ROOT tree

    max_elements : int (default=4)
      maximum number of elements to load from jagged arrays
    """   
    self._max_elements = max_elements
    self._df = pd.DataFrame(index=range(self.tree.num_entries))

    for branch in branches:
      self._join_branch(branch)

    return self._set_columns_names(self._df)


  def _join_branch(self, branch):
    """joins a branch to self._df"""
    df = self.tree.arrays(branch, library="pd")

    if len(df) > self.tree.num_entries:
      self._add_jagged_branch(df, branch)
    else:
      self._add_branch(df, branch)


  def _add_branch(self, df, branch: str):
    """adds a non-jagged branch to self.dataframe"""
    self._df[branch] = self.tree[branch].array(library="pd").values


  def _add_jagged_branch(self, df, branch):
    """adds a jagged branch to self.dataframe"""
    df = df.unstack().iloc[:,:self._max_elements]
    df.columns = [f"{branch}{i}" for i in range(self._max_elements)]
    self._df = self._df.join(df)

  @staticmethod
  def _set_columns_names(df):
    df.columns = df.columns.str.lower().str.replace(".","_")
    return df