"""
pipeline.py - the occlusion-aware synthetic-aperture pipeline, end to end.

    photos + poses
      -> colour-free foliage masks for every view   (planesweep.foreground_masks)
      -> background depth of the reference view      (render.background_sweep)
      -> all-in-focus leaf-free image                (render.render_all_in_focus)
      -> colour-free fruit detection and 3-D count   (detect.detect_fruit)

Every step is driven by geometry: depth from multi-view photo-consistency
decides what is foliage, which photos see each point and where the fruit is.

Dependencies: numpy, opencv-python.
"""

import time

import cv2
import numpy as np

from . import planesweep, render, sgm


DEFAULT_CFG = dict(
    z_near=0.40, z_far=1.40,      # search range of the first sweep [m]
    mask_scale=0.5,               # resolution of the foliage-mask sweep (fraction of processing size)
    mask_neighbours=6, mask_win=5, mask_keep=0.5, mask_step_px=2.0, mask_dilate_px=2,
    z_split=None,                 # foliage/background split depth; None = Otsu (automatic)
    bg_step_px=1.0, bg_win_detect=3, bg_win_render=7, bg_tau=0.15,
    wall_front=0.07,              # fruit can hang at most this far in front of the wall [m]
    wall_back=0.05,               # background search beyond the wall [m]
    use_sgm=True,                 # semi-global aggregation of the background cost volume
    half_select=True,             # left/right half-aperture selection against half-occlusion
    sgm_P1=0.08, sgm_P2=0.6, sgm_alpha=8.0, sgm_pre_win=3,
)


def background_range(depth_ref, z_split, front, back):
    """
    Depth window of the background sweep.

    Fruit hangs on the wall, so the background sweep only needs to search from
    `front` metres in front of the wall to `back` metres behind it.  The wall
    depth is the median of the reference depths beyond the foliage split
    (the fence and backdrop fill most of the view).  This physical prior stops
    weakly observed pixels (few visible views) from settling on spurious depths
    far in front of the wall, which would otherwise look like protruding fruit.
    """
    bg = depth_ref[depth_ref > z_split]
    z_wall = float(np.median(bg)) if bg.size >= 100 else 1.5 * z_split
    return max(z_split * 1.01, z_wall - front), z_wall + back, z_wall


def deocclude(rig, cfg=None, verbose=True):
    """
    Run the occlusion-removal stages on a rig.

    Returns dict with:
      masks        per-view foliage masks (processing resolution, 1 = foliage)
      z_split      foliage/background split depth [m]
      bg_range     (near, far) of the background sweep [m]
      depth        per-pixel background depth of the reference view [m] (small window,
                   used for detection); depth_render: larger window, used for the image
      cost         photo-consistency cost at that depth
      confidence   1 - cost_min / cost_mean over the sweep (0 = flat cost curve, ambiguous)
      image        leaf-free all-in-focus reference image (float32 BGR)
      support      views used per pixel (0 = unobserved)
      timings      seconds per stage
    """
    c = dict(DEFAULT_CFG)
    if cfg:
        c.update(cfg)
    T = {}
    t0 = time.time()
    fm = planesweep.foreground_masks(rig, c["z_near"], c["z_far"], mask_scale=c["mask_scale"],
                                     n_neighbours=c["mask_neighbours"], win=c["mask_win"],
                                     keep=c["mask_keep"], step_px=c["mask_step_px"],
                                     z_split=c["z_split"], dilate_px=c["mask_dilate_px"],
                                     verbose=verbose)
    T["masks"] = time.time() - t0
    z_split = fm["z_split"]
    lo, hi, z_wall = background_range(fm["depths"][rig.ref], z_split, c["wall_front"], c["wall_back"])
    f = rig.views[rig.ref].K[0, 0]
    planes = planesweep.inverse_depth_planes(lo, hi, c["bg_step_px"], f, rig.baselines().max())
    if verbose:
        print(f"    wall {z_wall:.3f} m; background sweep {lo:.3f}-{hi:.3f} m, {len(planes)} planes")
    t0 = time.time()
    if c["use_sgm"]:
        # guide for the edge-aware jump penalty: visibility-masked mean refocused on the wall
        guide_img, _ = render.refocus_plane(rig, z_wall, "mean", fm["masks"])
        guide = cv2.cvtColor(np.clip(guide_img, 0, 1), cv2.COLOR_BGR2GRAY)
        vol_all, vol_sel, supv = render.background_cost_volume(
            rig, fm["masks"], planes, tau=c["bg_tau"], pre_win=c["sgm_pre_win"],
            half_select=c["half_select"])
        T["bg_sweep"] = time.time() - t0
        t0 = time.time()
        # detection depth: half-aperture selected cost (sharp fruit edges, no side smear)
        S = sgm.sgm_aggregate(vol_sel, guide, c["sgm_P1"], c["sgm_P2"], c["sgm_alpha"])
        depth, cost, conf = sgm.wta_parabola(S, planes)
        # render depth: full-aperture cost (uses every view, best image quality)
        if vol_all is vol_sel:
            depth_render = depth
        else:
            S = sgm.sgm_aggregate(vol_all, guide, c["sgm_P1"], c["sgm_P2"], c["sgm_alpha"])
            depth_render, _, _ = sgm.wta_parabola(S, planes)
        T["sgm"] = time.time() - t0
        del S, vol_all, vol_sel, supv
    else:
        wd, wr = c["bg_win_detect"], c["bg_win_render"]
        bg = render.background_sweep(rig, fm["masks"], planes, wins=tuple(sorted({wd, wr})), tau=c["bg_tau"])
        T["bg_sweep"] = time.time() - t0
        depth, depth_render = bg[wd]["depth"], bg[wr]["depth"]
        cost, conf = bg[wd]["cost"], bg[wd]["confidence"]
    t0 = time.time()
    img, sup = render.render_all_in_focus(rig, depth_render, fm["masks"])
    T["render"] = time.time() - t0
    return dict(masks=fm["masks"], mask_depths=fm["depths"], z_split=z_split, bg_range=(lo, hi),
                z_wall=z_wall, depth=depth, depth_render=depth_render, cost=cost,
                confidence=conf, image=img, support=sup, timings=T)
