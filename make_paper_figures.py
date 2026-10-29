"""
Collect every figure of the paper into figures/, under the file names used
by the manuscript's \\includegraphics.

Multi-panel figures (Fig1-Fig5, app1) are assembled from the standalone
panels in */plots/ with a small LaTeX standalone document (requires
pdflatex with the `standalone` class); each composite is scaled to the page
width of the version used in the manuscript, so it appears at the same size.
Single-file appendix figures are copied as they are.

Run after the per-folder plotting scripts:
    python3 edmd/paper_edmd_figures.py
    python3 krylov/plot_section3_smooth_distances.py
    python3 convergence/plot_appendix.py
    python3 misspecification/plot_theta_landscape.py
    python3 noise/plot_noise_robustness.py
    python3 make_paper_figures.py
"""

import os
import shutil
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "figures")


def p(folder, name):
    return os.path.join(HERE, folder, "plots", name + ".pdf")


# name -> (panels, composite page width in pt)
COMPOSITES = {
    "Fig1": ([p("edmd", "fixed_basis_transform_heatmap_ccd"),
              p("edmd", "fixed_basis_transform_heatmap_dsa"),
              p("edmd", "fixed_basis_scatter_ccd"),
              p("edmd", "fixed_basis_scatter_dsa")], 768.451),
    "Fig2": ([p("edmd", "state_transform_nrmse"),
              p("edmd", "state_transform_pearson"),
              p("edmd", "state_transform_recovered_theta")], 595.276),
    "Fig3": ([p("edmd", "cross_basis_heatmap_ccd"),
              p("edmd", "cross_basis_heatmap_dsa"),
              p("edmd", "cross_basis_scatter_ccd"),
              p("edmd", "cross_basis_scatter_dsa")], 793.701),
    "Fig4": ([p("edmd", "basis_change_nrmse"),
              p("edmd", "basis_change_pearson")], 396.85),
    "Fig5": ([p("krylov", "smooth_bijection_deviation"),
              p("krylov", "smooth_bijection_n_and_a_sweeps"),
              p("krylov", "smooth_bijection_rank_sweep")], 595.276),
}

# name -> (source, page width in pt or None to copy unchanged)
SINGLES = {
    "app1": (p("edmd", "within_basis_8_heatmaps_rowwise"), 1365.32),
    "appendix_residual": (p("convergence", "appendix_residual"), None),
    "appendix_theta_landscape": (p("misspecification", "appendix_theta_landscape"), None),
    "appendix_noise_robustness": (p("noise", "appendix_noise_robustness"), None),
}

PANEL_HEIGHT = "4.5cm"
LABELS = "abcdefgh"


def compile_tex(body, width_pt, out_pdf):
    tex = (
        "\\documentclass[border=1pt]{standalone}\n"
        "\\usepackage{graphicx}\n"
        "\\begin{document}\n"
        f"\\resizebox{{{width_pt}pt}}{{!}}{{%\n{body}}}\n"
        "\\end{document}\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "fig.tex"), "w") as f:
            f.write(tex)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig.tex"],
                       cwd=tmp, check=True, stdout=subprocess.DEVNULL)
        shutil.copy(os.path.join(tmp, "fig.pdf"), out_pdf)


def composite_body(panels):
    cols = "@{}" + "@{\\hspace{8pt}}".join("c" for _ in panels) + "@{}"
    graphics = " & ".join(f"\\includegraphics[height={PANEL_HEIGHT}]{{{f}}}" for f in panels)
    labels = " & ".join(f"\\sffamily\\bfseries\\Large ({LABELS[i]})" for i in range(len(panels)))
    return f"\\begin{{tabular}}{{{cols}}}\n{graphics} \\\\[2pt]\n{labels}\n\\end{{tabular}}%\n"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    missing = [f for panels, _ in COMPOSITES.values() for f in panels]
    missing += [src for src, _ in SINGLES.values()]
    missing = [f for f in missing if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError("Run the plotting scripts first; missing:\n  " + "\n  ".join(missing))

    have_latex = shutil.which("pdflatex") is not None
    if not have_latex:
        print("pdflatex not found: composites skipped, single figures copied.")

    for name, (panels, width) in COMPOSITES.items():
        if have_latex:
            out = os.path.join(OUT_DIR, f"{name}.pdf")
            compile_tex(composite_body(panels), width, out)
            print(f"Saved {out}")

    for name, (src, width) in SINGLES.items():
        out = os.path.join(OUT_DIR, f"{name}.pdf")
        if width is not None and have_latex:
            compile_tex(f"\\includegraphics{{{src}}}%\n", width, out)
        else:
            shutil.copy(src, out)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
