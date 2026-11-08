"""Draw panel (a) of figures/capture_setup.png (the capture rig) in TikZ and paste it over the
old panel, keeping panel (b), the photo, and the image size unchanged.

    python3 make_setup_panel.py     (needs pdflatex, pdftoppm and Pillow)
"""
import math
import pathlib
import random
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE.parent / 'capture_setup.png'
rng = random.Random(7)
PW, PH = 9.12, 4.30          # panel size in cm; 912 x 430 px at 254 dpi


def vine_leaf(cx, cy, s, rot, fill):
    """Grapevine leaf: five palmate lobes, serrated margin, petiole sinus at the base."""
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


def bunch(cx, cy, r, rows):
    out = [f'\\draw[stem, line width=0.5pt] ({cx:.2f},{cy + r * 1.5:.2f}) -- ({cx + 0.03:.2f},{cy + r * 3:.2f});']
    for i, n in enumerate(rows):
        for j in range(n):
            out.append(f'\\shade[ball color=grapered] ({cx + (j - (n - 1) / 2) * 1.85 * r:.3f},{cy - i * 1.6 * r:.3f}) circle ({r:.3f});')
    return '\n'.join(out)


# box geometry (oblique view): front face, top offset
x0, y0, x1, y1 = 1.0, 1.25, 5.5, 3.55
dx, dy = 0.95, 0.55
hx0, hy0, hx1, hy1 = 1.75, 1.7, 4.75, 3.15           # viewing hole
scene = []
scene.append(f'\\fill[interior] ({hx0},{hy0}) rectangle ({hx1},{hy1});')
scene.append(f'\\fill[backdrop, opacity=0.9] ({hx0 + 0.25},{hy0 + 0.2}) rectangle ({hx1 - 0.15},{hy1 - 0.08});')
scene.append(f'\\fill[lit, opacity=0.35] ({hx0 + 1.2},{hy0}) -- ({hx1},{hy0}) -- ({hx1},{hy1}) -- ({hx0 + 2.2},{hy1}) -- cycle;')
for cx, cy in [(2.25, 2.75), (2.9, 2.5), (3.55, 2.85), (4.2, 2.55), (2.6, 2.2), (3.95, 2.15)]:
    scene.append(bunch(cx, cy, 0.055, [4, 3, 3, 2, 1]))
for cx in [2.1, 2.75, 3.4, 4.05, 4.6]:
    scene.append(f'\\draw[stem, line width=0.6pt] ({cx},{hy1}) .. controls ({cx + 0.15},{hy1 - 0.5}) and ({cx - 0.1},{hy1 - 0.9}) .. ({cx + 0.05},{hy0 + 0.1});')
for cx, cy, s, rot in [(2.0, 2.95, 0.3, 0.2), (2.55, 2.55, 0.34, -0.3), (3.2, 2.95, 0.3, 0.4), (3.75, 2.4, 0.36, 0.1),
                       (4.4, 2.9, 0.32, -0.4), (2.25, 2.0, 0.3, 0.5), (3.05, 1.95, 0.33, -0.2), (4.35, 2.0, 0.3, 0.3),
                       (4.65, 2.45, 0.25, -0.6)]:
    scene.append(vine_leaf(cx, cy, s, rot, rng.choice(['leafA', 'leafB', 'leafC'])))

tex = r'''\documentclass[tikz,border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\definecolor{bg}{HTML}{F3EEF9}
\definecolor{card}{HTML}{C8A167}\definecolor{cardtop}{HTML}{D8B47C}\definecolor{cardside}{HTML}{8C6A3C}
\definecolor{interior}{HTML}{2B2116}\definecolor{backdrop}{HTML}{5E8F64}\definecolor{lit}{HTML}{FFF2B0}
\definecolor{leafA}{HTML}{4E8A3A}\definecolor{leafB}{HTML}{5F9C45}\definecolor{leafC}{HTML}{3F7531}
\definecolor{grapered}{HTML}{A3243B}\definecolor{stemc}{HTML}{6B4E2E}
\definecolor{railc}{HTML}{E9DCC0}\definecolor{steel}{HTML}{9AA0A6}\definecolor{axis}{HTML}{2F74C0}
\tikzset{stem/.style={stemc}, leafedge/.style={black!60!leafC}, leafvein/.style={white!70!leafA},
         lab/.style={font=\scriptsize, anchor=west, text=black!80}, lead/.style={black!45, line width=0.35pt}}
\begin{document}
\begin{tikzpicture}
\useasboundingbox (0,0) rectangle (''' + f'{PW},{PH}' + r''');
\fill[bg] (0,0) rectangle (''' + f'{PW},{PH}' + r''');
\begin{scope}[shift={(0.05,0.3)}, scale=0.8]
% table
\fill[black!6] (0,0.15) -- (6.4,0.15) -- (7.3,1.55) -- (0.9,1.55) -- cycle;
% box: closed, front face with a viewing hole
\fill[cardtop] (''' + f'{x0},{y1}) -- ({x1},{y1}) -- ({x1 + dx},{y1 + dy}) -- ({x0 + dx},{y1 + dy}' + r''') -- cycle;
\fill[cardside] (''' + f'{x1},{y0}) -- ({x1 + dx},{y0 + dy}) -- ({x1 + dx},{y1 + dy}) -- ({x1},{y1}' + r''') -- cycle;
\fill[card] (''' + f'{x0},{y0}) rectangle ({x1},{y1}' + r''');
\draw[black!35, line width=0.3pt] (''' + f'{x0},{y1}) -- ({x1},{y1}) -- ({x1},{y0}) ({x1},{y1}) -- ({x1 + dx},{y1 + dy})' + r''';
\draw[black!20, line width=0.3pt] (''' + f'{x0 + dx / 2},{y1 + dy / 2}) -- ({x1 + dx / 2},{y1 + dy / 2})' + r''';
''' + '\n'.join(scene) + r'''
\draw[black!55, line width=0.5pt] (''' + f'{hx0},{hy0}) rectangle ({hx1},{hy1}' + r''');
% rail
\fill[railc] (0.35,0.55) -- (6.3,0.55) -- (6.55,0.95) -- (0.6,0.95) -- cycle;
\draw[steel, line width=0.6pt] (0.45,0.68) -- (6.38,0.68) (0.53,0.82) -- (6.47,0.82);
\foreach \x in {0.9,1.3,...,6.1} \draw[black!30, line width=0.25pt] (\x,0.62) -- (\x+0.04,0.66);
% lamp behind the rail (viewer side), shining into the scene
\fill[lit, opacity=0.45] (5.0,1.5) -- (''' + f'{hx0 + 1.3},{hy1 - 0.05}) -- ({hx1 - 0.05},{hy0 + 0.05}' + r''') -- cycle;
\fill[black!70] (6.05,0.22) ellipse (0.32 and 0.08);
\draw[black!75, line width=1.1pt] (6.05,0.25) -- (6.25,1.05) -- (5.35,1.55);
\fill[black!80] (5.0,1.47) -- (5.32,1.78) -- (5.55,1.42) -- (5.25,1.25) -- cycle;
\fill[lit] (5.08,1.42) circle (0.07);
% portrait iPhone on its holder, back towards the viewer, camera facing the box
\fill[black!55] (2.95,0.82) rectangle (3.25,0.98);
\fill[black!75] (3.04,0.98) rectangle (3.16,1.12);
\fill[black!85, rounded corners=1.5pt] (2.93,1.1) rectangle (3.27,1.78);
\fill[black!60, rounded corners=1pt] (2.97,1.55) rectangle (3.11,1.73);
\fill[black!20] (3.015,1.68) circle (0.025) (3.065,1.6) circle (0.025) (3.015,1.6) circle (0.02);
\draw[axis, densely dashed, line width=0.5pt] (3.1,1.62) -- (3.25,2.45);
% labels
\draw[lead] (5.9,3.6) -- (6.75,3.95); \node[lab] at (6.8,3.95) {closed cardboard box};
\draw[lead] (4.75,3.0) -- (6.75,3.45); \node[lab] at (6.8,3.45) {viewing hole};
\draw[lead] (4.45,2.55) -- (6.75,2.95); \node[lab, align=left] at (6.8,2.88) {grapevine leaves in\\front of red grapes};
\draw[lead] (5.3,1.6) -- (6.75,2.25); \node[lab, align=left] at (6.8,2.18) {lamp behind the rail,\\lighting the scene};
\draw[lead] (3.27,1.4) -- (6.75,1.6); \node[lab, align=left] at (6.8,1.53) {iPhone, portrait,\\on the rail};
\draw[lead] (3.25,2.45) -- (3.6,3.9) -- (6.75,4.3); \node[lab, text=axis] at (6.8,4.3) {optical axis};
\draw[lead] (6.4,0.8) -- (6.75,0.75); \node[lab] at (6.8,0.75) {rail, 20\,mm stops};
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
orig = HERE / 'capture_setup_original.png'
if not orig.exists():
    Image.open(FIG).save(orig)
fig = Image.open(orig).convert('RGB')
fig.paste(panel, (0, 0))
fig.save(FIG)
for ext in ('aux', 'log'):
    (HERE / f'setup_panel.{ext}').unlink(missing_ok=True)
print('capture_setup.png updated', fig.size)
