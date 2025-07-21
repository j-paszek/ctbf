# CTBF: Cancer Tree Biopsy Framework

### Authors: Jarosław Paszek, Agnieszka Mykowiecka, Krzysztof Gogolewski

Presented on ISMB/ECCB 2025.

_____

Detailed dscriptions of the project is available here:

[ISMBECCB2025_Long_Abstract_1886](https://j-paszek.github.io/ctbf/ISMBECCB2025_Long_Abstract_1886.pdf)

Graphical overview as a poster:

[ISMBECCB2025_1938_Paszek_poster](https://j-paszek.github.io/ctbf/ISMBECCB2025_1938_Paszek_poster.pdf)

_____

To run CTBF simply run:
`python ctbs.py`

Important note, to run pairwise CNP comparisons we use [cnp2cnp](https://github.com/AEVO-lab/cnp2cnp) tool.
We encourage you to download the tool and set paths `cnp2cnp_FOLDER` and `cnp2cnp_FILE` in `ctbs.py`.

_____

Repository contains:
- source code files:
  - `ctbs.py` - main file
  - `simulator.py` - simulates an evolution of a cancer cell, enables to take biopsy samples, see `Evolution Model` section for more information
  - `reconstructor.py` - reconstructs cancer tree from biopsy samples, see `Tree Reconstruction` section for more information
  - `evaluator.py` - computes similarity between two cancer trees, see `Tree Comparison` section for more information 
- config files:
  - `config_for_pic.json`

https://j-paszek.github.io/ctbf/simulated_tree.html

https://j-paszek.github.io/ctbf/true_tree.html

https://j-paszek.github.io/ctbf/reconstructed.html

https://j-paszek.github.io/ctbf/nj.html