# CDFD

The code in CDFD computes Circular Directional Flow Decomposition of networks and their circularity/directionality. This is an implementation of the methods presented in the paper *[in preparation]*.

## Background

The code in this repository implements algorithms to quantify the circularity and directedness of weighted directed networks. Our method is based on the decomposition (CDFD) of networks into an acyclic part and a perfectly cyclic one. Then, the proportion of the weight in the perfectly cyclic (respectively acyclic) part gives a measure of circularity (respectively directedness) of the network. This approach improves on previous notions of circularity/directedness in several ways as discussed in *[in preparation]*. 

In general there is more than one CDFD of a network, here we implement BFF and Maximal. The Maximal finds the maximum possible circularity in the given network, whereas the BFF locally fairly distributes the circularity between different edges. We believe that BFF is a better representative of a typical circular part of a decomposition, but one may wish to use the Maximal instead as its implementation is currently faster.

More technically we say that a weighted directed network is directional or acyclic if it does not contain any directed cycles. We say that a weighted directed network is circular if it can be expressed as a sum of balanced cycles (a balanced cycle is a directed cycle where all edges have the same weight). Then, a CDFD of a network consists on two subnetworks, a circular one and an acyclic one, that add up to the whole network. 

## Usage

```python
import networkx as nx
from decomposition_helpers import get_circularity, get_directedness, CDFD

# Graph under consideration
G = nx.DiGraph() 
G.add_weighted_edges_from([(1, 2, 1), (2, 3, 1), (1,3,1), (3,1,1)])

# returns: C is circular part and D the acyclic one using the BFF algorithm. 
C, D = CDFD(G, solution_method = "BFF")

# Return: value of the circularity ratio of the decomposition found above
get_circularity(C,D)

# returns: C is circular part and D the acyclic one using the Maximal algorithm. 
C, D = CDFD(G, solution_method = "BFF")

# Return: value of the directedness ratio of the decomposition found above
get_directedness(C,D)
```
Another example, with clear depictions of the different networks can be found in example_CDFD.ipynb. 

## Files

- **CDFD.py:** Definition of functions needed to find CDFD decompositions and circularity/directedness of a network. The main packages used are numpy (v. 1.24.3), networkx (v. 3.1), scipy (v. 1.12.0) and to a lesser degree pulp (v. 2.8.0) and ortools (v. 9.8.3296). 

- **example_CDFD.ipynb:** Example of usage of CDFD's main functions with depictions of the corresponding networks. 

- **other_helpers:** Supplementary file to compute other measures of circularity from the literature and to generate random networks.
  
- **generate_paper_figures** Directory with code necessary to generate the figures from the paper *[in preparation]*. The order of compilation must be:
  
    1. **data_sarafu:** Include in this directory the raw sarafu data. This data is available via the UK Data Service (UKDS) under their End User License, which stipulates suitable data-privacy protections. The dataset is available for download from the UKDS ReShare repository (https://reshare.ukdataservice.ac.uk/855142/) to users registered with the UKDS.
    2. **sarafu_data_processing.ipynb:** Cleans sarafu data.
    3. **generate_data_figures.ipynb:** Generates data to be depicted in figures. To decrese computational times some parameters have been reduced from the paper (but in comments we indicate the values used for the paper's figures). 
    4. **plot_figures.ipynb:** Generates the figures. 
