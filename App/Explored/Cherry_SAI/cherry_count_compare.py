#!/usr/bin/env python3
"""AMME5710: exploratory fruit counting after synthetic aperture refocusing.

Runs a reproducible, classical CV baseline across:
  1) the central original view,
  2) fruit-refocused mean/median/leaf-masked images,
  3) all original views registered at the measured fruit depth.

The counts are *automated detections*, NOT verified ground truth. Dark purple
fruit are especially difficult and are shown separately as review candidates.
A rectangular scene ROI is deliberately configurable: it is a prototype region
of interest, NOT a cherry-specific hard-coded position.

Dependencies: numpy, opencv-python. No scipy, torch or network downloads needed.
Usage:
 python cherry_count_compare.py --images images.zip --sai sai_results_24mm --out fruit_count_results
 python cherry_count_compare.py --images images.zip --sai sai_results_24mm.zip --out fruit_count_results
 python cherry_count_compare.py --images images.zip --sai sai_results_24mm --roi 0.19 0.31 0.82 0.62
"""
import argparse
import csv
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT = ('.png','.jpg','.jpeg','.tif','.tiff')


def _image_from_bytes(blob):
    im=cv2.imdecode(np.frombuffer(blob,np.uint8),cv2.IMREAD_COLOR)
    if im is None: raise ValueError('Could not decode image')
    return im


def load_originals(source, target_shape):
    source=Path(source)
    if source.is_dir():
        paths=sorted(f for f in source.rglob('*') if f.suffix.lower() in IMAGE_EXT)
        images=[_image_from_bytes(p.read_bytes()) for p in paths]
        names=[p.name for p in paths]
    elif source.suffix.lower()=='.zip':
        with zipfile.ZipFile(source) as z:
            names=sorted(n for n in z.namelist() if n.lower().endswith(IMAGE_EXT))
            images=[_image_from_bytes(z.read(n)) for n in names]
    else: raise ValueError('--images must be a folder or .zip file')
    h,w=target_shape
    resized=[cv2.resize(im,(w,h),interpolation=cv2.INTER_AREA) for im in images]
    if len(resized)<3:raise ValueError('Need at least 3 camera views')
    return resized,names


def load_sai(source):
    source=Path(source)
    required=['reference.png','focus_fruit_mean.png','focus_fruit_median.png',
              'focus_fruit_leafmasked.png','diagnostics.json']
    if source.is_dir():
        blobs={p.name:p.read_bytes() for p in source.rglob('*') if p.is_file() and p.name in required}
    elif source.suffix.lower()=='.zip':
        with zipfile.ZipFile(source) as z:
            blobs={Path(n).name:z.read(n) for n in z.namelist() if Path(n).name in required}
    else:raise ValueError('--sai must be a folder or .zip file')
    missing=[name for name in required if name not in blobs]
    if missing:raise FileNotFoundError('SAI output missing: '+', '.join(missing))
    diag=json.loads(blobs['diagnostics.json'].decode('utf8'))
    imgs={k.replace('.png',''):_image_from_bytes(blobs[k]) for k in required if k.endswith('.png')}
    shapes={im.shape[:2] for im in imgs.values()}
    if len(shapes)!=1:raise ValueError('SAI images do not have matching sizes')
    return imgs,diag


def get_roi(shape,box):
    h,w=shape
    x0,y0,x1,y1=box
    if not(0<=x0<x1<=1 and 0<=y0<y1<=1):raise ValueError('ROI fractions must be within 0...1 in x0 y0 x1 y1 order')
    x0,x1=int(round(w*x0)),int(round(w*x1))
    y0,y1=int(round(h*y0)),int(round(h*y1))
    roi=np.zeros((h,w),np.uint8);roi[y0:y1,x0:x1]=255
    return roi,(x0,y0,x1,y1)


def primary_colour_mask(bgr,roi,a_min=140):
    """Conservative reddish/pink detection, separate from unreliable very dark candidates.

    Lab 'a' captures redness while R/G and R/B suppress brownish/neutral clutter.
    No specific object centre/position or fruit number is encoded.
    """
    lab=cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB)
    r=bgr[:,:,2].astype(np.float32);g=bgr[:,:,1].astype(np.float32);b=bgr[:,:,0].astype(np.float32)
    a=lab[:,:,1]
    colour=(a>=a_min)&(r>g*1.18+1)&(r>b*1.12+1)&(r>=26)&(roi>0)
    u=np.uint8(colour)*255
    u=cv2.morphologyEx(u,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    u=cv2.morphologyEx(u,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    return u


def labelled_colour_objects(bgr,roi,a_min=140,min_area=13,max_area=340):
    m=primary_colour_mask(bgr,roi,a_min)
    contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    output=[]
    for c in contours:
        area=float(cv2.contourArea(c)); x,y,w,h=cv2.boundingRect(c)
        if not(min_area<=area<=max_area) or w<3 or h<3 or max(w,h)/max(1,min(w,h))>2.4:continue
        per=cv2.arcLength(c,True)
        circularity=4*np.pi*area/(per*per) if per>0 else 0
        if circularity<.24:continue
        mm=cv2.moments(c)
        if mm['m00']<=0: continue
        cx=float(mm['m10']/mm['m00']);cy=float(mm['m01']/mm['m00'])
        output.append({'x':cx,'y':cy,'area':area,'radius':float(np.sqrt(area/np.pi)),
                       'circularity':float(circularity),'bbox':[x,y,w,h]})
    output.sort(key=lambda o:(o['y'],o['x']))
    return output,m


def low_conf_dark_candidates(im,roi,bright_objects,center_diff=13,min_radius=4.4):
    """Exploratory round dark blobs; expected false positives from stems/shadows.

    These are shown on a separate overlay and never silently included in
    the conservative fruit count.
    """
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    bg=cv2.medianBlur(gray,41)
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    green=cv2.inRange(hsv,(30,35,22),(105,255,255))>0
    mask=(bg.astype(np.int16)-gray.astype(np.int16)>center_diff)&(gray<78)&(~green)&(roi>0)
    dist=cv2.distanceTransform(np.uint8(mask),cv2.DIST_L2,5)
    local_max=cv2.dilate(dist,np.ones((11,11),np.uint8))
    ys,xs=np.where((dist>=min_radius)&(dist>=local_max-.01))
    lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB)
    dark=[]
    h,w=gray.shape
    for x,y in zip(xs,ys):
        d=float(dist[y,x]);rad=int(np.ceil(d+4))
        if x<rad or x>=w-rad or y<rad or y>=h-rad:continue
        if any((x-b['x'])**2+(y-b['y'])**2<(max(9,b['radius']+3))**2 for b in bright_objects):continue
        # Exclude most blue-black pipe cleaners, though very dark cherries remain challenging.
        patch=im[y-2:y+3,x-2:x+3]
        if np.mean(patch[:,:,2].astype(float)-patch[:,:,0].astype(float)) < -2.0:continue
        if np.mean(lab[y-2:y+3,x-2:x+3,1])<128:continue
        th=np.linspace(0,2*np.pi,20,endpoint=False)
        ringx=np.clip(np.rint(x+np.cos(th)*(d+3)).astype(int),0,w-1)
        ringy=np.clip(np.rint(y+np.sin(th)*(d+3)).astype(int),0,h-1)
        ring=gray[ringy,ringx].astype(np.float32)
        core=float(np.median(gray[y-2:y+3,x-2:x+3]))
        if np.mean(ring-core>11)<.60:continue
        dark.append({'x':float(x),'y':float(y),'radius':d,'contrast':float(np.mean(ring)-core)})
    # Suppress multiple local maxima from a single large dark object
    dark.sort(key=lambda p:-p['contrast'])
    final=[]
    for ob in dark:
        if not any(np.hypot(ob['x']-p['x'],ob['y']-p['y'])<9 for p in final):final.append(ob)
    return sorted(final,key=lambda p:(p['y'],p['x']))


def overlay(im,objects,roi_rect,title,color=(30,220,255),dark=None,show_roi=True):
    out=im.copy();x0,y0,x1,y1=roi_rect
    if show_roi:cv2.rectangle(out,(x0,y0),(x1,y1),(120,205,235),1)
    for i,ob in enumerate(objects,1):
        p=(int(round(ob['x'])),int(round(ob['y'])))
        rr=max(5,int(round(ob.get('radius',6)))+3)
        cv2.circle(out,p,rr,color,2,lineType=cv2.LINE_AA)
        cv2.putText(out,str(i),(p[0]+rr+1,p[1]-2),cv2.FONT_HERSHEY_SIMPLEX,.37,color,1,cv2.LINE_AA)
    if dark is not None:
        for ob in dark:
            p=(int(ob['x']),int(ob['y']))
            cv2.circle(out,p,int(max(5,round(ob['radius']))),(255,220,0),1,lineType=cv2.LINE_AA)
    cv2.rectangle(out,(0,0),(min(out.shape[1],650),38),(18,18,18),-1)
    cv2.putText(out,title,(12,26),cv2.FONT_HERSHEY_SIMPLEX,.63,(255,255,255),2,cv2.LINE_AA)
    return out


def fuse_multi_view(originals,diag,roi,roi_rect,a_min,min_support,spacing=None,depth=None):
    N=len(originals);h,w=originals[0].shape[:2]
    if N!=diag['image_count_total']:
        raise ValueError(f'Number of input photographs ({N}) differs from SAI diagnostic ({diag["image_count_total"]})')
    ref=int(diag['reference_image_1based'])-1
    if not(0<=ref<N):raise ValueError('Invalid reference index')
    oldw=diag['image_size_original'][0]
    fx=float(diag['fx_original_px_used'])*w/oldw
    B=float(spacing if spacing is not None else diag['camera_step_mm'])
    Z=float(depth if depth is not None else diag['fruit_depth_mm'])
    offsets=np.asarray(diag['vertical_offsets_px_at_processing_scale'],np.float32)
    dwidth=diag['image_size_processed'][0]
    offsets=offsets*w/dwidth
    vote=np.zeros((h,w),np.uint8)
    sources=[]
    ones=np.ones((h,w),np.uint8)*255
    for i,im in enumerate(originals):
        # Detect in raw input, then map the candidate pixels to the fruit plane.
        objects,m=labelled_colour_objects(im,np.ones((h,w),np.uint8)*255,a_min)
        M=np.float32([[1,0,fx*B*(i-ref)/Z], [0,1,-(offsets[i]-offsets[ref])]])
        shifted=cv2.warpAffine(m,M,(w,h),flags=cv2.INTER_NEAREST,borderValue=0)
        shifted=cv2.bitwise_and(shifted,roi)
        # Dilating by 1 pixel accommodates interpolation and small calibration errors.
        shifted=cv2.dilate(shifted,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        vote+=(shifted>0).astype(np.uint8)
        sources.append({'image_index':i+1,'filename':None,'raw_candidates':len(objects),
                        'registered_mask_pixels':int(np.count_nonzero(shifted))})
    fused=np.uint8(vote>=min_support)*255
    fused=cv2.morphologyEx(fused,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    counts,labels,stats,centroids=cv2.connectedComponentsWithStats(fused,8)
    out=[]
    for i in range(1,counts):
        x,y,ww,hh,area=stats[i];cx,cy=centroids[i]
        if not(12<=area<=350) or ww<3 or hh<3 or max(ww,hh)/max(1,min(ww,hh))>2.8:continue
        comp=(labels[y:y+hh,x:x+ww]==i).astype(np.uint8)
        cc,_=cv2.findContours(comp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        perimeter=cv2.arcLength(max(cc,key=cv2.contourArea),True) if cc else 0
        circ=4*np.pi*area/(perimeter*perimeter) if perimeter else 0
        if circ <.28:continue
        support=int(np.max(vote[labels==i]));weighted_support=float(np.mean(vote[labels==i]))
        out.append({'x':float(cx),'y':float(cy),'area':float(area), 'radius':float(np.sqrt(area/np.pi)),
                    'support_peak':support,'support_mean':round(weighted_support,2),
                    'circularity':float(circ), 'bbox':[int(x),int(y),int(ww),int(hh)]})
    out.sort(key=lambda o:(o['y'],o['x']))
    return out,vote,fused,sources


def track_multiview(originals,diag,roi,roi_rect,a_min,min_support,spacing=None,depth=None):
    """Experimental per-instance tracking with nearest-neighbour epipolar geometry.

    Uses the rail parallax model to centre observations in a common reference view.
    A small residual velocity is learned per track to accommodate depth variation.
    This is a baseline tracker, not globally optimised 3D triangulation.
    """
    h,w=originals[0].shape[:2];N=len(originals)
    ref=diag['reference_image_1based']-1
    fx=diag['fx_original_px_used']*w/diag['image_size_original'][0]
    B=spacing if spacing is not None else diag['camera_step_mm']
    Z=depth if depth is not None else diag['fruit_depth_mm']
    vert=np.asarray(diag['vertical_offsets_px_at_processing_scale'])*w/diag['image_size_processed'][0]
    tracks=[]
    raw_dets=[]
    for i,im in enumerate(originals):
        objects,_=labelled_colour_objects(im,np.ones((h,w),np.uint8)*255,a_min)
        detections=[]
        for ob in objects:
            x=ob['x']+fx*B*(i-ref)/Z
            y=ob['y']-(vert[i]-vert[ref])
            xi,yi=int(round(x)),int(round(y))
            if not(0<=xi<w and 0<=yi<h and roi[yi,xi]>0):continue
            detections.append({'x':x,'y':y,'frame':i,'area':ob['area'],'radius':ob['radius']})
        raw_dets.append(detections)
        pairs=[]
        for tr_idx,tr in enumerate(tracks):
            last=tr[-1];gap=i-last['frame']
            if gap>4:continue
            velx,vely=0.,0.
            if len(tr)>=2:
                prev=tr[-2];d=max(1,last['frame']-prev['frame'])
                velx=np.clip((last['x']-prev['x'])/d,-6,6)
                vely=np.clip((last['y']-prev['y'])/d,-4,4)
            predx=last['x']+velx*gap; predy=last['y']+vely*gap
            for d_idx,detection in enumerate(detections):
                xy=np.hypot(detection['x']-predx,detection['y']-predy)
                lim=8.0+4.0*(gap-1)
                if xy<lim:
                    pairs.append((xy,tr_idx,d_idx))
        pairs.sort()
        used_d=set();used_t=set()
        for cost,tidx,didx in pairs:
            if tidx in used_t or didx in used_d:continue
            tracks[tidx].append(detections[didx]);used_t.add(tidx);used_d.add(didx)
        for j,d in enumerate(detections):
            if j not in used_d:tracks.append([d])
    output=[]
    for tr in tracks:
        if len(tr)<min_support:continue
        # Estimate position at reference frame by linear regression across viewpoints.
        frames=np.array([p['frame'] for p in tr],dtype=float)
        xx=np.array([p['x'] for p in tr]); yy=np.array([p['y'] for p in tr])
        if np.ptp(frames)>=2:
            x=float(np.polyval(np.polyfit(frames,xx,1),ref))
            y=float(np.polyval(np.polyfit(frames,yy,1),ref))
            predx=np.polyval(np.polyfit(frames,xx,1),frames)
            predy=np.polyval(np.polyfit(frames,yy,1),frames)
            residual=float(np.sqrt(np.mean((predx-xx)**2+(predy-yy)**2)))
        else:
            x=float(np.median(xx));y=float(np.median(yy));residual=0.
        if not (0<=round(x)<w and 0<=round(y)<h and roi[int(round(y)),int(round(x))]):continue
        if residual>9:continue
        output.append({'x':x,'y':y,'radius':float(np.median([p['radius'] for p in tr])),
                       'area':float(np.median([p['area'] for p in tr])),
                       'support_peak':len(tr),'support_mean':round(residual,2),
                       'circularity':0,'bbox':[0,0,0,0],
                       'confidence_type':'multiview_track_unverified'})
    output.sort(key=lambda o:(o['y'],o['x']))
    return output,raw_dets


def write_csv(path,objects):
    keys=['id','x_px','y_px','radius_px','area_px2','circularity','support_peak','support_mean','confidence_type']
    with open(path,'w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for i,o in enumerate(objects,1):
            writer.writerow({'id':i,'x_px':round(o['x'],2),'y_px':round(o['y'],2),
                             'radius_px':round(o.get('radius',0),2),'area_px2':round(o.get('area',0),2),
                             'circularity':round(o.get('circularity',0),2),
                             'support_peak':o.get('support_peak',''),
                             'support_mean':o.get('support_mean',''),
                             'confidence_type':o.get('confidence_type','colour_candidate')})


def make_comparison(source_images,outdir,rect):
    panels=[]
    x0,y0,x1,y1=rect
    crop=(max(0,x0-8),max(0,y0-10),min(source_images[0][1].shape[1],x1+8),
          min(source_images[0][1].shape[0],y1+10))
    for title,img in source_images:
        panel=img[crop[1]:crop[3],crop[0]:crop[2]]
        panel=cv2.resize(panel,(680,445),interpolation=cv2.INTER_AREA)
        canvas=np.full((490,680,3),245,dtype=np.uint8);canvas[45:]=panel
        cv2.putText(canvas,title,(14,30),cv2.FONT_HERSHEY_SIMPLEX,.72,(30,30,30),2,cv2.LINE_AA)
        panels.append(canvas)
    cv2.imwrite(str(outdir/'count_comparison.jpg'),np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])]))


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images',default='images.zip',help='Original 12 photographs, zip or folder')
    ap.add_argument('--sai',default='sai_results_24mm',help='Previous SAI results folder or ZIP')
    ap.add_argument('--out',default='fruit_count_results')
    ap.add_argument('--roi',type=float,nargs=4,default=(.20,.325,.82,.61),metavar=('X0','Y0','X1','Y1'),
                    help='Analysis ROI, fractions of reference width/height (prototype only)')
    ap.add_argument('--lab-a-min',type=int,default=140,help='Colour redness threshold: lower increases recall and false positives')
    ap.add_argument('--min-support',type=int,default=2,help='Number of original views required for fused colour mask')
    ap.add_argument('--track-support',type=int,default=4,help='Minimum matched viewpoints for exploratory per-instance tracks (default 4)')
    ap.add_argument('--true-count',type=int,default=None,help='Optional manually verified physical fruit total, to calculate counting errors')
    ap.add_argument('--spacing',type=float,default=None,help='Override camera spacing in mm')
    ap.add_argument('--depth',type=float,default=None,help='Override shared registration depth in mm')
    args=ap.parse_args()
    if args.min_support<1:ap.error('--min-support must be >=1')
    if not 120<=args.lab_a_min<=180:ap.error('--lab-a-min expected range 120-180')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    imgs,diag=load_sai(args.sai)
    h,w=imgs['reference'].shape[:2]
    print(f'Loaded SAI results ({w}x{h}), reference image #{diag["reference_image_1based"]}')
    roi,rect=get_roi((h,w),args.roi)
    variants=[('Original reference','reference'),('SAI mean','focus_fruit_mean'),
              ('SAI median','focus_fruit_median'),('SAI leaf masked','focus_fruit_leafmasked')]
    results={};annotated={}
    for title,k in variants:
        objs,mask=labelled_colour_objects(imgs[k],roi,args.lab_a_min)
        for o in objs:o['confidence_type']='colour_candidate'
        dark=low_conf_dark_candidates(imgs[k],roi,objs) if k in ('reference','focus_fruit_leafmasked') else []
        results[k]={'count_bright_colour_candidates':len(objs),'count_dark_review_candidates':len(dark)}
        write_csv(out/(k+'_detections.csv'),objs)
        cv2.imwrite(str(out/(k+'_mask.png')),mask)
        overlay_img=overlay(imgs[k],objs,rect,f'{title}: {len(objs)} red/pink candidates',dark=dark)
        cv2.imwrite(str(out/(k+'_annotated.png')),overlay_img)
        annotated[k]=overlay_img
        if dark:
            for o in dark:o['confidence_type']='unverified_dark_blob'
            write_csv(out/(k+'_dark_review.csv'),dark)
        print(f'  {title:<22} red/pink={len(objs):2d}, extra DARK review candidates={len(dark):2d}')
    originals,names=load_originals(args.images,(h,w))
    fused,votes,fused_mask,per_view=fuse_multi_view(originals,diag,roi,rect,args.lab_a_min,args.min_support,
                                                    args.spacing,args.depth)
    for i,d in enumerate(per_view):d['filename']=names[i]
    for o in fused:o['confidence_type']='multi_view_colour_candidate'
    results['multi_view']={'count_bright_colour_candidates':len(fused),
                           'minimum_support_views':args.min_support}
    write_csv(out/'multi_view_detections.csv',fused)
    heat=cv2.applyColorMap(np.uint8(votes*255/max(1,len(originals))),cv2.COLORMAP_TURBO)
    heat=np.where((roi>0)[...,None],heat,imgs['reference']//3)
    cv2.imwrite(str(out/'multi_view_support.png'),heat)
    cv2.imwrite(str(out/'multi_view_mask.png'),fused_mask)
    mv=overlay(imgs['reference'],fused,rect,
               f'Multi-view fused: {len(fused)} red/pink candidates',color=(30,255,30))
    cv2.imwrite(str(out/'multi_view_annotated.png'),mv)
    tracks,raw_dets=track_multiview(originals,diag,roi,rect,args.lab_a_min,args.track_support,
                                     args.spacing,args.depth)
    # Merge near-identical track estimates; keep the track observed in more frames.
    # NOT a complete solution to correct object identity.
    tracks.sort(key=lambda ob:-ob['support_peak'])
    nms_tracks=[]
    for ob in tracks:
        if not any(np.hypot(ob['x']-kept['x'],ob['y']-kept['y'])<8 for kept in nms_tracks):
            nms_tracks.append(ob)
    tracks=sorted(nms_tracks,key=lambda ob:(ob['y'],ob['x']))
    for t in tracks:t['confidence_type']='multi_view_track_unverified'
    write_csv(out/'multi_view_track_detections.csv',tracks)
    tracked_img=overlay(imgs['reference'],tracks,rect,
                        f'Multi-view tracks: {len(tracks)} red/pink candidates',color=(20,230,230))
    cv2.imwrite(str(out/'multi_view_tracks_annotated.png'),tracked_img)
    results['multi_view_tracks']={'count_bright_colour_candidates':len(tracks),
                                  'minimum_views_per_track':args.track_support,
                                  'method':'per-view colour detections; sequential geometric association'}
    print(f'  Multi-view tracks      red/pink={len(tracks):2d}')
    print(f'  Multi-view fused       red/pink={len(fused):2d}')
    make_comparison([('Reference',annotated['reference']),('SAI leaf-masked',annotated['focus_fruit_leafmasked']),
                     ('SAI median',annotated['focus_fruit_median']),('Multi-view tracks',tracked_img)],out,rect)
    summary={'purpose':'Unverified fruit detection comparison; NOT ground-truth counting accuracy',
             'parameters':{'roi_fractions':list(args.roi),'roi_pixels':list(rect),
                           'lab_a_min':args.lab_a_min,'min_support':args.min_support,'track_support':args.track_support,
                           'camera_spacing_mm':args.spacing or diag['camera_step_mm'],
                           'fruit_depth_mm_assumed':args.depth or diag['fruit_depth_mm'],
                           'focal_length_source':diag['fx_source'],
                           'focal_length_px':diag['fx_original_px_used'],
                           'reference_image_1based':diag['reference_image_1based']},
             'counts':results,'per_original_image':per_view,
             'limitations':[
               'Counts are candidate detections, not ground truth and cannot be called accurate yet.',
               'The prototype region of interest is scene-specific and manually configurable.',
               'Dark purple cherries can be missed; dark review candidates include false positives.',
               'Closely touching cherries can merge into a single connected component.',
               'The 3D fusion assumes all fruit lie near one depth plane; depth variability causes duplicates/misses.',
               'Synthetic aperture may improve visibility without increasing the number of colour-only detections.',
               'This is not yet object tracking or full instance segmentation.']}
    if args.true_count is not None:
        if args.true_count<=0: ap.error('--true-count must be positive')
        summary['ground_truth_fruit_count']=args.true_count
        summary['counting_errors']={k:{'detected':v['count_bright_colour_candidates'],
                                      'absolute_count_error':abs(v['count_bright_colour_candidates']-args.true_count),
                                      'signed_count_error':v['count_bright_colour_candidates']-args.true_count,
                                      'absolute_percentage_error':round(100*abs(v['count_bright_colour_candidates']-args.true_count)/args.true_count,1)}
                                   for k,v in results.items()}
    (out/'count_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    (out/'README_RESULTS.txt').write_text(
        'AMME5710 synthetic aperture cherry counting baseline\n\n'
        'Open count_comparison.jpg first.\n'
        'Red/pink numbered circles: automated LAB candidate fruit regions.\n'
        'Small cyan circles (where present): UNVERIFIED dark blob candidates.\n'
        'They are excluded from the reported conservative counts because stems and shadows are similar.\n'
        'Green numbered circles: strict fused pixel masks across original images.\n'
        'Yellow numbered circles: exploratory per-instance tracks across original images.\n'
        'Review multi_view_support.png for view agreement.\n\n'
        'The counts are not accuracy results; a physically verified fruit total is needed.\n'
        'Tune --lab-a-min or --min-support / --roi; compare to manual annotation.\n',encoding='utf8')
    print('\nSaved:',out/'count_comparison.jpg')
    print('IMPORTANT: Detection counts are unverified and may miss dark or touching cherries.')

if __name__=='__main__':main()
