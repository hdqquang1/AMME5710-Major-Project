#!/usr/bin/env python3
"""Manual point annotation/validation for AMME5710 cherry counting.

This tool creates labelled fruit centres for a single 2D image. Its 2D recall
metrics are not the same as counting every physical cherry in the model.

Example:
 python cherry_review_labels.py --image sai_results_24mm/focus_fruit_leafmasked.png \\
   --detections fruit_count_results/focus_fruit_leafmasked_detections.csv \\
   --out fruit_count_results/manual_cherries.csv

Left click: add a cherry centre. Right click: remove the closest centre.
S: save points and evaluation. U: undo. Q / Esc: exit.
"""
import argparse,csv,json
from pathlib import Path
import cv2
import numpy as np


def read_detections(path):
    if not path:return []
    with open(path,newline='',encoding='utf8') as f:
        return [(float(row['x_px']),float(row['y_px'])) for row in csv.DictReader(f)]


def match_points(preds,truth,r=12):
    poss=[]
    for a,p in enumerate(preds):
        for b,t in enumerate(truth):
            d=np.hypot(p[0]-t[0],p[1]-t[1])
            if d<=r:poss.append((d,a,b))
    matched_pred=set();matched_truth=set()
    for _,a,b in sorted(poss):
        if a not in matched_pred and b not in matched_truth:
            matched_pred.add(a);matched_truth.add(b)
    tp=len(matched_pred);fp=len(preds)-tp;fn=len(truth)-tp
    return {'ground_truth_visible_fruit':len(truth),'prediction_count':len(preds),
            'true_positive':tp,'false_positive':fp,'false_negative':fn,
            'precision':round(tp/(tp+fp),4) if tp+fp else None,
            'recall':round(tp/(tp+fn),4) if tp+fn else None,
            'matching_tolerance_px':r}


def save_points(path,truth,preds,tol):
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['id','x_px','y_px'])
        for i,(x,y) in enumerate(truth,1):w.writerow([i,x,y])
    metrics=match_points(preds,truth,tol)
    path.with_suffix('.metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--image',default='sai_results_24mm/focus_fruit_leafmasked.png')
    p.add_argument('--detections',default='fruit_count_results/focus_fruit_leafmasked_detections.csv')
    p.add_argument('--out',default='fruit_count_results/manual_cherries.csv')
    p.add_argument('--tolerance',type=float,default=12)
    args=p.parse_args()
    im=cv2.imread(str(args.image))
    if im is None:raise FileNotFoundError(args.image)
    preds=read_detections(args.detections)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    truth=[];history=[]
    if out.exists():
        with out.open(newline='',encoding='utf8') as f:
            truth=[(int(float(row['x_px'])),int(float(row['y_px']))) for row in csv.DictReader(f)]
        print(f'Restored {len(truth)} manual labels from {out}')
    h,w=im.shape[:2]
    window='Mark cherries | left=add right=remove s=save u=undo q=quit'
    cv2.namedWindow(window,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window,min(w,950),min(h,850))
    def callback(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN:
            history.append(truth.copy());truth.append((x,y))
        elif event==cv2.EVENT_RBUTTONDOWN and truth:
            ind=int(np.argmin([(x-a)**2+(y-b)**2 for a,b in truth]))
            history.append(truth.copy());truth.pop(ind)
    cv2.setMouseCallback(window,callback)
    while True:
        canvas=im.copy()
        for i,(x,y) in enumerate(preds,1):
            cv2.circle(canvas,(round(x),round(y)),7,(0,215,255),1,cv2.LINE_AA)
        for i,(x,y) in enumerate(truth,1):
            cv2.circle(canvas,(x,y),9,(40,255,40),2,cv2.LINE_AA)
            cv2.putText(canvas,str(i),(x+11,y-2),cv2.FONT_HERSHEY_SIMPLEX,.40,(40,255,40),1,cv2.LINE_AA)
        cv2.rectangle(canvas,(0,0),(min(w,710),35),(20,20,20),-1)
        cv2.putText(canvas,f'Manual: {len(truth)} | Auto: {len(preds)} | S=save U=undo Q=quit',
                    (10,24),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow(window,canvas)
        key=cv2.waitKey(70)&0xFF
        if key in [ord('s'),ord('S')]:
            metrics=save_points(out,truth,preds,args.tolerance)
            print('Saved',out, '\nMetrics:',metrics)
        elif key in [ord('u'),ord('U')] and history: truth[:]=history.pop()
        elif key in [ord('q'),ord('Q'),27]:break
    cv2.destroyAllWindows()
    print('Use S to save before exiting. Physical fruit count may differ from visible fruit labels.')

if __name__=='__main__':main()
