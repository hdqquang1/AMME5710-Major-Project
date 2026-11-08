"""Draw panel (a) of figures/capture_setup.png (the capture rig) in TikZ and paste it over the
old panel, keeping panel (b), the photo, and the image size unchanged.

Oblique view of a closed box: a large viewing hole (about 3 cm border at box scale), a slit across
the top near the back (2 cm from each side) through which the vines hang, grape clusters near the
back wall, leafy vines near the front, a phone on a stand level with the hole, the rail well in
front of the box and a lamp on the viewer's side lighting the scene.

    python3 make_setup_panel.py     (needs pdflatex, pdftoppm and Pillow)
"""
import math
import pathlib
import random
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE.parent / 'capture_setup.png'
rng = random.Random(11)
PW, PH = 9.12, 4.30

X0, Y0, X1, Y1 = 0.9, 1.9, 4.4, 4.0           # front face (about 1 unit = 17 cm)
DX, DY = 1.0, 0.6                              # depth direction
B = 0.17                                       # 3 cm border
HX0, HY0, HX1, HY1 = X0 + B, Y0 + B, X1 - B, Y1 - B
S = 0.12                                       # 2 cm slit inset
F = 0.85                                       # slit position along the depth


def P(x, y, d):
    return x + d * DX, y + d * DY


def vine_leaf(cx, cy, s, rot, fill):
    pts = []
    for k in range(360):
        t = 2 * math.pi * k / 360
        lobes = 0.78 + 0.22 * math.cos(5 * (t - math.pi / 2))
        teeth = 0.045 * abs(math.sin(22 * t))
        sinus = 1 - 0.55 * math.exp(-((t - 3 * math.pi / 2) / 0.22) ** 2)
        r = s * (lobes + teeth) * sinus * (0.85 if math.sin(t) < 0 else 1.0)
        pts.append((cx + r * math.cos(t + rot), cy + r * math.sin(t + rot)))
    path = ' -- '.join(f'({x:.3f},{y:.3f})' for x, y in pts)
    out = [f'\\fill[{fill}] {path} -- cycle;', f'\\draw[leafedge, line width=0.25pt] {path} -- cycle;']
    for k in range(5):
        t = math.pi / 2 + 2 * math.pi * (k - 2) / 5 * 0.8
        out.append(f'\\draw[leafvein, line width=0.3pt] ({cx:.3f},{cy - 0.25 * s:.3f}) -- '
                   f'({cx + 0.8 * s * math.cos(t + rot):.3f},{cy + 0.8 * s * math.sin(t + rot):.3f});')
    return '\n'.join(out)


def cluster(cx, cy, r, length, shade):
    """A natural bunch: a tapering cone of jittered, overlapping berries drawn back to front."""
    berries = []
    n_rows = int(length / (1.25 * r))
    for i in range(n_rows):
        frac = i / max(n_rows - 1, 1)
        width = (1 - 0.75 * frac) * 5.0 * r * (1.0 if frac > 0.12 else 0.8)
        n = max(1, round(width / (1.4 * r)))
        for j in range(n):
            x = cx + (j - (n - 1) / 2) * 1.4 * r + rng.uniform(-0.35, 0.35) * r + 0.25 * r * math.sin(3 * frac)
            y = cy - i * 1.25 * r + rng.uniform(-0.25, 0.25) * r
            berries.append((y, x, r * rng.uniform(0.85, 1.1)))
    berries.sort(reverse=True)
    out = [f'\\draw[stem, line width=0.45pt] ({cx:.3f},{cy + 0.2:.3f}) -- ({cx:.3f},{cy:.3f});']
    out += [f'\\shade[ball color={shade}] ({x:.3f},{y:.3f}) circle ({rr:.3f});' for y, x, rr in berries]
    return '\n'.join(out)


inside = []
bx0, by0 = P(X0, Y0, 1)
bx1, by1 = P(X1, Y1, 1)
inside.append(f'\\fill[wallL] ({X0},{Y0}) -- ({bx0},{by0}) -- ({bx0},{by1}) -- ({X0},{Y1}) -- cycle;')
inside.append(f'\\fill[floorc] ({X0},{Y0}) -- ({X1},{Y0}) -- ({bx1},{by0}) -- ({bx0},{by0}) -- cycle;')
inside.append(f'\\fill[ceilc] ({X0},{Y1}) -- ({X1},{Y1}) -- ({bx1},{by1}) -- ({bx0},{by1}) -- cycle;')
inside.append(f'\\fill[backwall] ({bx0},{by0}) rectangle ({bx1},{by1});')
sx0, sy = P(X0 + S, Y1, F)
sx1, _ = P(X1 - S, Y1, F)
inside.append(f'\\fill[lit, opacity=0.35] ({sx0:.2f},{sy:.2f}) -- ({sx1:.2f},{sy:.2f}) -- ({sx1 - 0.1:.2f},{Y0 + 0.3:.2f}) -- ({sx0 - 0.1:.2f},{Y0 + 0.3:.2f}) -- cycle;')
inside.append(f'\\fill[lit, opacity=0.30] ({HX0 + 1.2},{HY0}) -- ({HX1 + 0.8},{HY0 + 0.4}) -- ({HX1 + 0.8},{HY1}) -- ({HX0 + 2.4},{HY1}) -- cycle;')
for u in [0.12, 0.3, 0.48, 0.66, 0.84]:
    x, y = P(X0 + S + u * (X1 - X0 - 2 * S), Y1, F)
    hang = rng.uniform(0.75, 1.05)
    inside.append(f'\\draw[stem, line width=0.55pt] ({x:.3f},{y:.3f}) .. controls ({x + 0.05:.3f},{y - 0.4:.3f}) '
                  f'and ({x - 0.05:.3f},{y - 0.8:.3f}) .. ({x + 0.02:.3f},{y - hang:.3f});')
    inside.append(cluster(x + 0.02, y - hang + 0.05, 0.058, rng.uniform(0.5, 0.65), rng.choice(['grapeA', 'grapeB'])))
for u, drop in [(0.04, 1.4), (0.42, 0.75), (0.97, 1.2)]:
    x = HX0 + u * (HX1 - HX0)
    y = HY1 + 0.05
    inside.append(f'\\draw[stem, line width=0.8pt] ({x:.3f},{y:.3f}) .. controls ({x + 0.12:.3f},{y - 0.45:.3f}) '
                  f'and ({x - 0.12:.3f},{y - 0.9:.3f}) .. ({x + 0.04:.3f},{y - drop:.3f});')
    for k in range(2):
        f = (k + 0.7) / 2.4
        inside.append(vine_leaf(x + rng.uniform(-0.18, 0.18), y - f * drop, rng.uniform(0.16, 0.2),
                                rng.uniform(-0.6, 0.6), rng.choice(['leafA', 'leafB', 'leafC'])))

tex = r'''\documentclass[tikz,border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\definecolor{bg}{HTML}{F3EEF9}
\definecolor{card}{HTML}{C8A167}\definecolor{cardtop}{HTML}{D8B47C}\definecolor{cardside}{HTML}{8C6A3C}
\definecolor{wallL}{HTML}{6E5534}\definecolor{floorc}{HTML}{8A6C43}\definecolor{ceilc}{HTML}{4A3A24}
\definecolor{backwall}{HTML}{6F9E73}\definecolor{lit}{HTML}{FFF2B0}
\definecolor{leafA}{HTML}{4E8A3A}\definecolor{leafB}{HTML}{5F9C45}\definecolor{leafC}{HTML}{3F7531}
\definecolor{grapeA}{HTML}{A3243B}\definecolor{grapeB}{HTML}{8A1E3A}\definecolor{stemc}{HTML}{6B4E2E}
\definecolor{railc}{HTML}{E9DCC0}\definecolor{steel}{HTML}{9AA0A6}\definecolor{axis}{HTML}{2F74C0}
\tikzset{stem/.style={stemc}, leafedge/.style={black!60!leafC}, leafvein/.style={white!70!leafA},
         lab/.style={font=\scriptsize, anchor=west, text=black!80}, lead/.style={black!45, line width=0.35pt}}
\begin{document}
\begin{tikzpicture}
\useasboundingbox (0,0) rectangle (''' + f'{PW},{PH}' + r''');
\fill[bg] (0,0) rectangle (''' + f'{PW},{PH}' + r''');
\begin{scope}[shift={(0.05,0.05)}, scale=0.74]
\fill[black!6] (-0.1,0.05) -- (6.3,0.05) -- (7.0,2.6) -- (0.6,2.6) -- cycle;
\fill[cardtop] (''' + f'{X0},{Y1}) -- ({X1},{Y1}) -- ({X1 + DX},{Y1 + DY}) -- ({X0 + DX},{Y1 + DY}' + r''') -- cycle;
\fill[cardside] (''' + f'{X1},{Y0}) -- ({X1 + DX},{Y0 + DY}) -- ({X1 + DX},{Y1 + DY}) -- ({X1},{Y1}' + r''') -- cycle;
\fill[card] (''' + f'{X0},{Y0}) rectangle ({X1},{Y1}' + r''');
\fill[black!80] (''' + f'{sx0:.3f},{sy:.3f}) -- ({sx1:.3f},{sy:.3f}) -- ({sx1 + 0.04:.3f},{sy + 0.025:.3f}) -- ({sx0 + 0.04:.3f},{sy + 0.025:.3f}' + r''') -- cycle;
\begin{scope}
\clip (''' + f'{HX0},{HY0}) rectangle ({HX1},{HY1}' + r''');
''' + '\n'.join(inside) + r'''
\end{scope}
\draw[black!55, line width=0.5pt] (''' + f'{HX0},{HY0}) rectangle ({HX1},{HY1}' + r''');
\draw[black!35, line width=0.3pt] (''' + f'{X0},{Y1}) -- ({X1},{Y1}) -- ({X1},{Y0}) ({X1},{Y1}) -- ({X1 + DX},{Y1 + DY})' + r''';
% rail: a drawer slide on a board standing upright on the table, well in front of the box
\fill[railc] (0.3,0.2) rectangle (5.9,1.5);
\fill[white!70!railc] (0.3,1.5) -- (5.9,1.5) -- (6.02,1.6) -- (0.42,1.6) -- cycle;
\fill[black!25!railc] (5.9,0.2) -- (6.02,0.3) -- (6.02,1.6) -- (5.9,1.5) -- cycle;
\fill[steel] (0.45,1.12) rectangle (5.75,1.36);
\draw[black!45, line width=0.3pt] (0.45,1.24) -- (5.75,1.24);
\foreach \x in {0.8,1.2,...,5.6} \fill[black!35] (\x,1.18) circle (0.012);
\fill[black!65] (2.3,1.1) rectangle (2.9,1.42);
% phone on a short holder on the carriage, camera facing the box
\fill[black!55] (2.47,1.6) rectangle (2.73,1.72);
\fill[black!85, rounded corners=1.5pt] (2.42,1.72) rectangle (2.78,2.42);
\fill[black!60, rounded corners=1pt] (2.46,2.19) rectangle (2.6,2.37);
\fill[black!20] (2.505,2.32) circle (0.025) (2.555,2.24) circle (0.025) (2.505,2.24) circle (0.02);
\draw[axis, densely dashed, line width=0.5pt] (2.53,2.27) -- (2.9,2.5);
% lamp at the middle of the rail, on the viewer's side, shining straight into the box
\fill[lit, opacity=0.45] (3.2,1.95) -- (''' + f'{HX0 + 0.6},{HY1 - 0.1}) -- ({HX1 - 0.3},{HY1 - 0.1}' + r''') -- cycle;
\fill[black!70] (3.35,0.08) ellipse (0.3 and 0.07);
\draw[black!75, line width=1.1pt] (3.35,0.1) -- (3.55,1.0) -- (3.25,1.8);
\fill[black!80] (3.0,1.78) -- (3.25,2.12) -- (3.5,1.9) -- (3.3,1.62) -- cycle;
\fill[lit] (3.18,1.95) circle (0.07);
\draw[lead] (2.9,2.5) -- (7.05,5.38); \node[lab, text=axis] at (7.1,5.38) {optical axis};
\draw[lead] ''' + f'({(sx0 + sx1) / 2 + 0.6:.2f},{sy + 0.01:.2f})' + r''' -- (7.05,4.69); \node[lab] at (7.1,4.69) {slit, 2\,cm from each side};
\draw[lead] (''' + f'{X1 + 0.5},{Y1 - 0.4}' + r''') -- (7.05,4.00); \node[lab] at (7.1,4.00) {closed cardboard box};
\draw[lead] (''' + f'{HX1},{HY1 - 0.15}' + r''') -- (7.05,3.31); \node[lab] at (7.1,3.31) {viewing hole, 3\,cm border};
\draw[lead] (''' + f'{HX1 - 0.3},{HY1 - 0.35}' + r''') -- (7.05,2.62); \node[lab] at (7.1,2.62) {grapes back, vines front};
\draw[lead] (3.45,1.95) -- (7.05,1.93); \node[lab] at (7.1,1.93) {lamp at the middle of the rail};
\draw[lead] (2.78,2.1) -- (7.05,1.24); \node[lab] at (7.1,1.24) {iPhone, portrait, on the rail};
\draw[lead] (5.9,0.8) -- (7.05,0.55); \node[lab] at (7.1,0.55) {upright rail, 20\,mm stops};
\end{scope}
\end{tikzpicture}
\end{document}
'''
(HERE / 'setup_panel.tex').write_text(tex)
subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'setup_panel.tex'], cwd=HERE, check=True,
               stdout=subprocess.DEVNULL)
subprocess.run(['pdftoppm', '-png', '-r', '254', '-singlefile', 'setup_panel.pdf', 'setup_panel'], cwd=HERE, check=True)
from PIL import Image  # noqa: E402
panel = Image.open(HERE / 'setup_panel.png').convert('RGB').resize((912, 430), Image.LANCZOS)
fig = Image.open(HERE / 'capture_setup_original.png').convert('RGB')
fig.paste(panel, (0, 0))
fig.save(FIG)
for ext in ('aux', 'log'):
    (HERE / f'setup_panel.{ext}').unlink(missing_ok=True)
print('capture_setup.png updated', fig.size)
