#!/bin/python


import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('PS') 
matplotlib.use('PS')

import daft

daft._rendering_context


# Instantiate the PGM.
pgm = daft.PGM([4.3, 4.05], origin=[0.3, 0.3], observed_style="shaded")

# Hierarchical parameters.
pgm.add_node(daft.Node("alpha", r"$\alpha$", 1, 4))
pgm.add_node(daft.Node("DP", r"$DP$", 2, 4, observed=True))
pgm.add_node(daft.Node("w_j", r"$w_j$", 2, 3))
pgm.add_node(daft.Node("c_i", r"$c_i$", 2, 2))
pgm.add_node(daft.Node("v_pi", r"$v_{pi}$", 2, 1, observed=True))
pgm.add_node(daft.Node("d_pi", r"$d_{pi}$", 1, 1, observed=True))
pgm.add_node(daft.Node("H", r"$H$", 3, 4, observed=True))
pgm.add_node(daft.Node("phi_pj", r"$\phi_{pj}$", 3, 3))

# Add in the edges.
pgm.add_edge("alpha", "DP")
pgm.add_edge("DP", "w_j")
pgm.add_edge("w_j", "c_i")
pgm.add_edge("c_i", "v_pi")
pgm.add_edge("d_pi", "v_pi")

pgm.add_edge("H", "phi_pj")
pgm.add_edge("phi_pj", "v_pi")

# Add plates.


pgm.add_plate(daft.Plate([1.5, 2.5, 2, 1], label=r"$j = 1, \cdots, k$",
        shift=-0.1))

pgm.add_plate(daft.Plate([0.4, 0.45, 3, 1.83], label=r"$i = 1, \cdots, n$",
        shift=-0.1, label_offset=[175, 0]))



# Render and save.
pgm.render()
pgm.figure.savefig("single_pgm.png")

