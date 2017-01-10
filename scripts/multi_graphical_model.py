#!/bin/python


import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('PS')
matplotlib.use('PS')

import daft


# Instantiate the PGM.
pgm = daft.PGM([4.3, 4.05], origin=[0.3, 0.3], observed_style="shaded")

# Hierarchical parameters.
pgm.add_node(daft.Node("alpha", r"$\alpha$", 1, 4))
pgm.add_node(daft.Node("DP", r"$DP$", 2, 4, observed=True))
pgm.add_node(daft.Node("pi_k", r"$\pi_k$", 2, 3))
pgm.add_node(daft.Node("z_n", r"$z_{n}$", 2, 2))
pgm.add_node(daft.Node("v_mn", r"$v_{mn}$", 2, 1, observed=True))
pgm.add_node(daft.Node("d_mn", r"$d_{mn}$", 1, 1, observed=True))
pgm.add_node(daft.Node("H", r"$H$", 3, 4, observed=True))
pgm.add_node(daft.Node("phi_mk", r"$\phi_{mk}$", 3, 3))

# Add in the edges.
pgm.add_edge("alpha", "DP")
pgm.add_edge("DP", "pi_k")
pgm.add_edge("pi_k", "z_n")
pgm.add_edge("z_n", "v_mn")
pgm.add_edge("d_mn", "v_mn")

pgm.add_edge("H", "phi_mk")
pgm.add_edge("phi_mk", "v_mn")

# Add plates.

pgm.add_plate(daft.Plate([1.55, 2.60, 2, 1], label=r"$k = 1, \cdots, K$",
        shift=-0.1, label_offset=[115, 0]))

pgm.add_plate(daft.Plate([0.4, 0.45, 3, 1.83], label=r"$n = 1, \cdots, N$",
        shift=-0.1, label_offset=[175, 0]))

pgm.add_plate(daft.Plate([2.55, 2.78, 0.85, 0.62], label=r"$m = 1, \cdots, M$",
        shift=-0.1, label_offset=[49, 20]))

pgm.add_plate(daft.Plate([0.62, 0.7, 1.75, 1], label=r"$m = 1, \cdots, M$",
        shift=-0.1, label_offset=[100,0]))


# Render and save.
pgm.render()
pgm.figure.savefig("multi_pgm.png")
pgm.figure.savefig("multi_pgm.pdf")


