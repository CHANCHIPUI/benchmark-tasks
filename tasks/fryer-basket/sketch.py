"""Draw side_view.png: a side view of the fryer (Avantco F100) through its centreline, from geometry.py, with the basket
down in the oil and hanging on the hanger edge, the hook in close-up, and the main heights.

    python sketch.py
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle

import geometry as g

L, W, H = g.BASKET
P = g.HANDLE_PITCH


def basket(ax, u_back, bottom, colour, label):
    fw = u_back - L
    ax.add_patch(Rectangle((fw * 100, bottom * 100), L * 100, H * 100, fill=False, ec=colour, lw=1.6, hatch="..",
                           label=label))
    top = bottom + H
    # riser and coated grip
    ax.plot([fw * 100, fw * 100], [top * 100, (top + g.HANDLE_STEM) * 100], color=colour, lw=1.6)
    ex = fw - g.HANDLE_LEN * math.cos(P)
    ez = top + g.HANDLE_STEM + g.HANDLE_LEN * math.sin(P)
    ax.plot([fw * 100, ex * 100], [(top + g.HANDLE_STEM) * 100, ez * 100], color="black", lw=4)
    # hook: tab back from the back wall's top, lip down
    ax.plot([u_back * 100, (u_back + g.HOOK_REACH) * 100, (u_back + g.HOOK_REACH) * 100],
            [top * 100, top * 100, (top - g.HOOK_LIP) * 100], color=colour, lw=2)
    # fries (three layers)
    for layer in range(3):
        for row in range(5):
            x = fw + 0.035 + row * 0.042 + 0.012 * layer
            z = bottom + g.WALL + layer * (g.FRY[2] + 0.001)
            ax.add_patch(Rectangle((x * 100 - 0.4, z * 100), 0.8, 0.8, color="#e8b84a"))


fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2.2, 1]})
for a in (ax, bx):
    # body below the deck (minus the tank), the head, oil, hanger lip
    a.add_patch(Rectangle((0, 0), g.DEPTH * 100, g.VAT_FLOOR * 100, color="#c9ccd0"))
    a.add_patch(Rectangle((0, g.VAT_FLOOR * 100), g.VAT_U[0] * 100, (g.DECK - g.VAT_FLOOR) * 100, color="#c9ccd0"))
    a.add_patch(Rectangle((g.HEAD_U * 100, g.VAT_FLOOR * 100), (g.DEPTH - g.HEAD_U) * 100, (g.HEAD_TOP - g.VAT_FLOOR) * 100,
                          color="#b9bcc0"))
    a.add_patch(Rectangle((g.HEAD_U * 100, (g.DECK + 0.03) * 100), 0.3, 6, color="#b02020"))
    a.add_patch(Rectangle((g.VAT_U[0] * 100, g.VAT_FLOOR * 100), (g.HEAD_U - g.VAT_U[0]) * 100,
                          (g.OIL - g.VAT_FLOOR) * 100, color="#d9a74a", alpha=0.35, label="oil (water in trials)"))
    a.add_patch(Circle((g.ROD_U * 100, g.ROD_Z * 100), g.ROD_R * 100, color="#555"))
    basket(a, g.REST_U, g.RACK, "#3b6ea8", "basket down in the oil")
    basket(a, g.HUNG_U, g.ROD_Z + g.ROD_R + g.HOOK_T - H, "#2a8a3b", "basket hung on the hanger")
    a.set_aspect("equal")
    a.set_xlabel("cm back from the fryer's front face")
    a.set_ylabel("cm above the table")
ax.set_xlim(-22, 44)
ax.set_ylim(0, 48)
ax.legend(loc="upper left", fontsize=8)
ax.set_title("Avantco F100 and 177FBASKET, side view through the centreline (nominal: the hung basket tilts 3-4 deg)", fontsize=9)
for z, lab in ((g.DECK, f"deck {g.DECK * 100:.1f}"), (g.OIL, f"oil {g.OIL * 100:.1f}"), (g.HEIGHT, f"head top {g.HEIGHT * 100:.1f}")):
    ax.axhline(z * 100, color="grey", lw=0.6, ls=":")
    ax.text(43.5, z * 100 + 0.4, lab, ha="right", fontsize=8, color="dimgrey")
bx.set_xlim(26, 34)
bx.set_ylim(27, 33)
bx.set_title("hook on the hanger edge (close-up)")
bx.text(26.3, 27.4, f"hanger: the head's rolled top front edge, {2 * g.ROD_R * 1000:.0f} mm round,\n"
        f"its top {(g.ROD_Z + g.ROD_R) * 100:.1f} cm up; hook tab {g.HOOK_REACH * 1000:.1f} mm, lip {g.HOOK_LIP * 1000:.0f} mm",
        fontsize=8)
fig.tight_layout()
fig.savefig("side_view.png", dpi=130)
print("side_view.png")
