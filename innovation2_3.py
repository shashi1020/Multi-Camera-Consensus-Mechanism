#!/usr/bin/env python3
"""
Adaptive Multi-Camera Consensus + Adversarial-Defense Framework
Safe research-only code: detects suspicious inputs, mitigates, logs examples, and optionally switches to a robust fallback model.
"""

import os
import json
import hashlib
import argparse
from datetime import datetime
import time
import random
from collections import deque, defaultdict

import numpy as np
import cv2
from ultralytics import YOLO

# -------------------------
# Config / Defaults (tune these)
# -------------------------
THREAT_SCORE_THRESHOLD = 0.55
THREAT_LOG_DIR = "threat_logs"
os.makedirs(THREAT_LOG_DIR, exist_ok=True)
FALLBACK_MAX_LATENCY = 0.6  # seconds allowed for robust model call (per-camera)
DEFAULT_CAM_PROBE_MAX = 4   # try indices 0..3 by default
DISPLAY_WINDOWS = True

# -------------------------
# Utilities: camera probe
# -------------------------
def find_working_cameras(max_indices=DEFAULT_CAM_PROBE_MAX, width=640, height=480):
    good_caps = []
    cam_indices = []
    for i in range(max_indices):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # warm-up read
        time.sleep(0.15)
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            continue
        good_caps.append(cap)
        cam_indices.append(i)
    return cam_indices, good_caps

# -------------------------
# Simple augment + draw utilities (from your code)
# -------------------------
def augment_image(image):
    flip = np.random.choice([True, False])
    angle = np.random.randint(-10, 10)
    brightness = np.random.uniform(0.5, 1.5)
    if flip:
        image = cv2.flip(image, 1)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
    return image

def draw_detections(img, detections, Box_colours, class_list):
    if img is None: return
    for det in detections:
        bb = det['bbox']
        clsID = det['class_id']
        conf = det['confidence']
        cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                      Box_colours[int(clsID)], 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"{class_list[int(clsID)]} {conf:.2f}",
                    (int(bb[0]), int(bb[1]) - 6), font, 0.6, (255, 0, 255), 2)

# -------------------------
# Consensus class (your IntelligentMultiCameraConsensus)
# -------------------------
class IntelligentMultiCameraConsensus:
    def __init__(self,
                 num_cameras,
                 camera_positions=None,
                 confidence_threshold=0.5,
                 center_distance_threshold=0.12,
                 temp_scale=1.2,
                 history_len=6,
                 min_persistence=2):
        self.num_cameras = num_cameras
        self.confidence_threshold = confidence_threshold
        self.camera_reliability_scores = np.ones(num_cameras, dtype=float)
        self.center_distance_threshold = center_distance_threshold
        self.temp_scale = temp_scale
        self.history = deque(maxlen=history_len)
        self.min_persistence = min_persistence
        self.camera_positions = camera_positions or {i: np.array([i, 0, 0], dtype=float) for i in range(num_cameras)}

    def process_detections(self, camera_detections, image_sizes):
        normalized = self._normalize_detections(camera_detections, image_sizes)
        self._calibrate_confidences(normalized)
        clusters = self._cluster_across_cameras(normalized)
        self._compute_spatial_support(clusters)
        consensus_list = self._make_consensus(clusters)
        validated = self._temporal_validation(consensus_list)
        self._update_camera_reliability(clusters, consensus_list)
        self.history.append(consensus_list)
        return consensus_list, validated

    def _normalize_detections(self, camera_detections, image_sizes):
        norm = {}
        for cam_id, dets in camera_detections.items():
            if cam_id not in image_sizes: continue
            w, h = image_sizes[cam_id]
            norm[cam_id] = []
            for d in dets:
                x1, y1, x2, y2 = d['bbox']
                cx = (x1 + x2) / 2.0 / w
                cy = (y1 + y2) / 2.0 / h
                nw = (x2 - x1) / w
                nh = (y2 - y1) / h
                norm[cam_id].append({
                    'orig': d,
                    'center': np.array([cx, cy]),
                    'size': np.array([nw, nh]),
                    'class_id': d['class_id'],
                    'confidence': d['confidence'],
                    'cam': cam_id
                })
        return norm

    def _calibrate_confidences(self, normalized):
        def temp_scale(conf):
            eps = 1e-6
            conf = np.clip(conf, eps, 1 - eps)
            logit = np.log(conf / (1 - conf))
            scaled = 1 / (1 + np.exp(-logit / self.temp_scale))
            return float(scaled)
        for cam_id, dets in normalized.items():
            for d in dets:
                d['confidence_cal'] = temp_scale(d['confidence'])

    def _cluster_across_cameras(self, normalized):
        clusters = []
        for cam_id, dets in normalized.items():
            for d in dets:
                placed = False
                for cl in clusters:
                    if d['class_id'] != cl['class_id']:
                        continue
                    centroid = cl['centroid']
                    dist = np.linalg.norm(d['center'] - centroid)
                    if dist <= self.center_distance_threshold:
                        cl['members'].append(d)
                        all_centers = np.array([m['center'] for m in cl['members']])
                        cl['centroid'] = np.mean(all_centers, axis=0)
                        placed = True
                        break
                if not placed:
                    clusters.append({'class_id': d['class_id'], 'members': [d], 'centroid': d['center'].copy()})
        return clusters

    def _compute_spatial_support(self, clusters):
        for cl in clusters:
            members = cl['members']
            n = len(members)
            if n <= 1:
                for m in members:
                    m['spatial_support'] = 0.0
                continue
            supports = []
            for i, mi in enumerate(members):
                s = 0.0
                for j, mj in enumerate(members):
                    if i == j: continue
                    center_dist = np.linalg.norm(mi['center'] - mj['center'])
                    center_score = np.exp(- (center_dist / (self.center_distance_threshold + 1e-6))**2)
                    size_diff = np.linalg.norm(mi['size'] - mj['size'])
                    size_score = 1.0 / (1.0 + size_diff*10.0)
                    pos_i = self.camera_positions[mi['cam']]
                    pos_j = self.camera_positions[mj['cam']]
                    cam_dist = np.linalg.norm(pos_i - pos_j)
                    geom_factor = 1.0 / (1.0 + cam_dist)
                    corr = center_score * size_score * geom_factor
                    s += corr
                s /= max(1, n-1)
                supports.append(s)
            for m, s in zip(members, supports):
                m['spatial_support'] = float(s)

    def _make_consensus(self, clusters):
        consensus_list = []
        for cl in clusters:
            members = cl['members']
            weights = []
            for m in members:
                cam_rel = float(self.camera_reliability_scores[m['cam']])
                weight = m.get('confidence_cal', m['confidence']) * cam_rel * (1.0 + m.get('spatial_support', 0.0))
                weights.append(weight)
            weights = np.array(weights) + 1e-8
            final_conf = float(np.sum([m.get('confidence_cal', m['confidence']) * w for m, w in zip(members, weights)]) / np.sum(weights))
            avg_sp = float(np.mean([m.get('spatial_support', 0.0) for m in members]))
            uncertainty = float(1.0 / (1.0 + avg_sp))
            consensus_strength = float(np.std(weights))
            per_camera_bboxes = defaultdict(lambda: None)
            for m in members:
                per_camera_bboxes[m['cam']] = m['orig']['bbox']
            consensus_list.append({
                'class_id': cl['class_id'],
                'final_confidence': final_conf,
                'uncertainty': uncertainty,
                'consensus_strength': consensus_strength,
                'members': members,
                'per_camera_bboxes': per_camera_bboxes,
            })
        return consensus_list

    def _temporal_validation(self, consensus_list):
        validated = []
        current_signatures = []
        for item in consensus_list:
            centroid = np.mean([m['center'] for m in item['members']], axis=0)
            current_signatures.append((item['class_id'], centroid))
        for idx, (cls, cent) in enumerate(current_signatures):
            count = 1
            for past in self.history:
                for p in past:
                    if p['class_id'] != cls:
                        continue
                    p_cent = np.mean([m['center'] for m in p['members']], axis=0)
                    if np.linalg.norm(p_cent - cent) <= self.center_distance_threshold:
                        count += 1
                        break
            if count >= self.min_persistence:
                validated.append(consensus_list[idx])
        return validated

    def _update_camera_reliability(self, clusters, consensus_list):
        decay = 0.995
        self.camera_reliability_scores *= decay
        for item in consensus_list:
            score = item['final_confidence'] * (1.0 - item['uncertainty'])
            for m in item['members']:
                cam = m['cam']
                self.camera_reliability_scores[cam] += 0.01 * score
        self.camera_reliability_scores = np.clip(self.camera_reliability_scores, 0.2, 2.0)

# -------------------------
# Threat assessor + mitigator + logging + model switcher
# -------------------------
def is_blurry(image, threshold=100.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold, fm

def brightness_level(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(hsv[...,2].mean())

def hash_image(image):
    h = hashlib.sha256(image.tobytes()).hexdigest()[:10]
    return h

class ThreatAssessor:
    def __init__(self, weight_uncertainty=0.5, weight_disagreement=0.3, weight_image_quality=0.2):
        self.wu = weight_uncertainty
        self.wd = weight_disagreement
        self.wi = weight_image_quality

    def compute_disagreement(self, consensus_list, per_camera_detections):
        if len(consensus_list) == 0: return 0.0
        disagreements = []
        num_cams = max(1, len(per_camera_detections))
        for item in consensus_list:
            cams_present = sum(1 for c in item['per_camera_bboxes'].keys() if item['per_camera_bboxes'][c] is not None)
            frac_missing = 1.0 - (cams_present / num_cams)
            disagreements.append(frac_missing)
        return float(np.mean(disagreements)) if disagreements else 0.0

    def compute_threat(self, consensus_list, per_camera_detections, frames):
        if len(consensus_list) == 0:
            avg_unc = 0.0
        else:
            avg_unc = float(np.mean([it.get('uncertainty', 1.0) for it in consensus_list]))
        disagreement = self.compute_disagreement(consensus_list, per_camera_detections)
        blur_vals = []
        bright_vals = []
        for cam, im in frames.items():
            if im is None: continue
            blurry, fm = is_blurry(im)
            blur_vals.append(1.0 - min(fm/200.0, 1.0))
            bright_vals.append(abs(brightness_level(im) / 255.0 - 0.5) * 2.0)
        image_quality_score = float(np.mean(blur_vals + bright_vals)) if (blur_vals or bright_vals) else 0.0
        threat_score = (self.wu * avg_unc) + (self.wd * disagreement) + (self.wi * image_quality_score)
        threat_score = float(np.clip(threat_score, 0.0, 1.0))
        meta = {'avg_uncertainty': avg_unc, 'disagreement': disagreement, 'image_quality': image_quality_score,
                'per_camera_blur': blur_vals, 'per_camera_brightness_extremes': bright_vals}
        return threat_score, meta

def jpeg_recompress(image, quality=70):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    if not result:
        return image
    decimg = cv2.imdecode(encimg, 1)
    return decimg

def median_denoise(image, ksize=3):
    return cv2.medianBlur(image, ksize)

def test_time_augment_ensemble(model, image, augment_fns=None, conf=0.45):
    if augment_fns is None:
        augment_fns = [lambda x: x, lambda x: cv2.flip(x, 1), lambda x: cv2.GaussianBlur(x, (3,3), 0)]
    all_preds = []
    for fn in augment_fns:
        im2 = fn(image)
        det_raw = model.track(source=[im2], conf=conf, save=False)
        if len(det_raw[0]) > 0:
            for b in det_raw[0].boxes:
                bb = b.xyxy.numpy()[0].copy()
                confv = float(b.conf.numpy()[0])
                cls = int(b.cls.numpy()[0])
                all_preds.append({'bbox': bb, 'confidence': confv, 'class_id': cls})
    merged = []
    used = [False]*len(all_preds)
    def iou(a,b):
        xA = max(a[0], b[0]); yA = max(a[1], b[1])
        xB = min(a[2], b[2]); yB = min(a[3], b[3])
        interW = max(0, xB-xA); interH = max(0, yB-yA)
        if interW==0 or interH==0: return 0.0
        inter = interW*interH
        areaA = (a[2]-a[0])*(a[3]-a[1]); areaB = (b[2]-b[0])*(b[3]-b[1])
        union = areaA + areaB - inter
        return inter / union if union>0 else 0.0
    for i,p in enumerate(all_preds):
        if used[i]: continue
        group = [p]; used[i]=True
        for j,q in enumerate(all_preds[i+1:], start=i+1):
            if used[j]: continue
            if p['class_id'] != q['class_id']: continue
            if iou(p['bbox'], q['bbox']) > 0.5:
                group.append(q); used[j] = True
        avg_conf = float(np.mean([g['confidence'] for g in group]))
        avg_box = np.mean(np.array([g['bbox'] for g in group]), axis=0)
        merged.append({'bbox': avg_box.tolist(), 'confidence': avg_conf, 'class_id': group[0]['class_id']})
    return merged

class ModelSwitcher:
    def __init__(self, fast_model, robust_model=None):
        self.fast = fast_model
        self.robust = robust_model
        self.fallback_count = 0

    def infer_raw(self, image, use_robust=False, conf=0.45):
        if use_robust and self.robust is not None:
            self.fallback_count += 1
            t0 = time.time()
            preds = self.robust.track(source=[image], conf=conf, save=False)
            latency = time.time() - t0
            return preds, latency, 'robust'
        else:
            t0 = time.time()
            preds = self.fast.track(source=[image], conf=conf, save=False)
            latency = time.time() - t0
            return preds, latency, 'fast'

    def infer_detection_dict(self, image, use_robust=False, conf=0.45):
        det_raw, latency, tag = self.infer_raw(image, use_robust=use_robust, conf=conf)
        dets = []
        if len(det_raw[0]) > 0:
            for b in det_raw[0].boxes:
                bb = b.xyxy.numpy()[0].copy()
                confv = float(b.conf.numpy()[0])
                cls = int(b.cls.numpy()[0])
                dets.append({'bbox': bb, 'confidence': confv, 'class_id': cls})
        return dets, latency, tag

class SuspiciousLogger:
    def __init__(self, outdir=THREAT_LOG_DIR):
        self.outdir = outdir
    def log(self, frames, consensus_list, meta, threat_score):
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")[:-3]
        entry = {'timestamp': ts, 'threat_score': float(threat_score), 'metadata': meta, 'consensus': []}
        entry_images = {}
        for cam_id, img in frames.items():
            if img is None: continue
            hsh = hash_image(img)
            filename = f"{ts}_cam{cam_id}_{hsh}.jpg"
            path = os.path.join(self.outdir, filename)
            cv2.imwrite(path, img)
            entry_images[str(cam_id)] = filename
        entry['images'] = entry_images
        for it in consensus_list:
            entry['consensus'].append({
                'class_id': int(it['class_id']),
                'final_confidence': float(it['final_confidence']),
                'uncertainty': float(it['uncertainty']),
                'consensus_strength': float(it['consensus_strength']),
                'per_camera_bboxes': {int(k): (v.tolist() if v is not None else None) for k,v in it['per_camera_bboxes'].items()}
            })
        outpath = os.path.join(self.outdir, f"{ts}_meta.json")
        with open(outpath, 'w') as fh:
            json.dump(entry, fh, indent=2)
        return outpath

# -------------------------
# Main runnable loop
# -------------------------
def main(args):
    # Load class list
    with open(args.class_list) as f:
        class_list = f.read().strip().split("\n")
    Box_colours = []
    for i in range(len(class_list)):
        Box_colours.append((random.randint(0,255), random.randint(0,255), random.randint(0,255)))

    # Load models
    print("Loading fast model:", args.fast_model)
    fast_model = YOLO(args.fast_model)
    robust_model = None
    if args.robust_model:
        try:
            print("Loading robust model:", args.robust_model)
            robust_model = YOLO(args.robust_model)
        except Exception as e:
            print("Could not load robust model:", e)
            robust_model = None

    # Build model switcher
    model_switcher = ModelSwitcher(fast_model=fast_model, robust_model=robust_model)
    threat_assessor = ThreatAssessor()
    slogger = SuspiciousLogger()

    # Camera or videos
    cam_indices = []
    caps = []
    if args.video_paths:
        # open listed videos
        for p in args.video_paths:
            cap = cv2.VideoCapture(p)
            if cap.isOpened():
                caps.append(cap)
                cam_indices.append(p)
    else:
        cam_indices, caps = find_working_cameras(max_indices=args.max_cam_probing)
        if len(caps) == 0:
            print("No cameras found. Exiting.")
            return

    num_cams = len(caps)
    camera_positions = {i: np.array([float(i), 0.0, 0.0]) for i in range(num_cams)}
    consensus_system = IntelligentMultiCameraConsensus(num_cameras=num_cams, camera_positions=camera_positions,
                                                      confidence_threshold=0.5, center_distance_threshold=0.12,
                                                      temp_scale=1.2, history_len=6, min_persistence=2)

    print(f"Using {num_cams} camera(s):", cam_indices)

    try:
        while True:
            frames = {}
            image_sizes = {}
            # read all frames
            for cam_pos, cap in enumerate(caps):
                ret, frame = cap.read()
                if not ret or frame is None:
                    frames[cam_pos] = None
                else:
                    frames[cam_pos] = frame
                    h, w = frame.shape[:2]
                    image_sizes[cam_pos] = (w, h)

            # if any camera dropped, break
            if all(fr is None for fr in frames.values()):
                print("All frames empty; ending.")
                break

            # Per-camera detection collect (use augmentation during inference lightly)
            camera_detections = {}
            for cam_pos, frame in frames.items():
                if frame is None:
                    camera_detections[cam_pos] = []
                    continue
                aug = augment_image(frame) if args.augment_inference else frame
                dets, lat, tag = model_switcher.infer_detection_dict(aug, use_robust=False, conf=args.conf)
                camera_detections[cam_pos] = dets

            # Visualize per-camera raw detections
            for cam_pos, frame in frames.items():
                if frame is None: continue
                draw_detections(frame, camera_detections[cam_pos], Box_colours, class_list)

            # Consensus
            consensus_list, validated = consensus_system.process_detections(camera_detections, image_sizes)

            # Overlay consensus info on frames
            for item in consensus_list:
                cls = item['class_id']; conf = item['final_confidence']; unc = item['uncertainty']
                for cam_id in range(num_cams):
                    frm = frames.get(cam_id)
                    if frm is None: continue
                    txt = f"C:{class_list[cls]} Conf:{conf:.2f} Unc:{unc:.2f}"
                    cv2.putText(frm, txt, (10, 30 + 20 * cam_id), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    if cam_id in item['per_camera_bboxes'] and item['per_camera_bboxes'][cam_id] is not None:
                        bb = item['per_camera_bboxes'][cam_id]
                        cv2.rectangle(frm, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0,255,0), 3)

            # Threat assessment
            threat_score, threat_meta = threat_assessor.compute_threat(consensus_list, camera_detections, frames)

            # If suspicious => mitigation + optional fallback consensus re-run
            if threat_score >= THREAT_SCORE_THRESHOLD:
                slog_path = slogger.log(frames, consensus_list, threat_meta, threat_score)
                # Preprocess frames quickly
                pre_frames = {cid: (jpeg_recompress(frames[cid], quality=70) if frames[cid] is not None else None) for cid in frames}
                # Try robust model per-camera if available and latency OK (measured)
                replaced_camera_dets = {}
                fallback_used = False
                if robust_model is not None:
                    latencies = []
                    for cam_pos, pf in pre_frames.items():
                        if pf is None:
                            replaced_camera_dets[cam_pos] = []
                            continue
                        dets, lat, tag = model_switcher.infer_detection_dict(pf, use_robust=True, conf=args.conf)
                        latencies.append(lat)
                        replaced_camera_dets[cam_pos] = dets
                    avg_lat = np.mean(latencies) if latencies else 0.0
                    # if robust latency acceptable, re-run consensus on robust results
                    if avg_lat <= FALLBACK_MAX_LATENCY:
                        fallback_used = True
                        # Draw overlay to indicate fallback
                        for cam_pos, frm in frames.items():
                            if frm is None: continue
                            cv2.putText(frm, f"MITIGATED: FALLBACK (score {threat_score:.2f})", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                        # update camera_detections and re-run consensus
                        camera_detections = replaced_camera_dets
                        consensus_list, validated = consensus_system.process_detections(camera_detections, image_sizes)
                    else:
                        # robust too slow — fallback to TTA per-camera (cheaper) on cam0..n
                        for cam_pos, pf in pre_frames.items():
                            if pf is None:
                                replaced_camera_dets[cam_pos] = []
                                continue
                            tta_dets = test_time_augment_ensemble(fast_model, pf, augment_fns=None, conf=args.conf)
                            replaced_camera_dets[cam_pos] = tta_dets
                        camera_detections = replaced_camera_dets
                        consensus_list, validated = consensus_system.process_detections(camera_detections, image_sizes)
                        for cam_pos, frm in frames.items():
                            if frm is None: continue
                            cv2.putText(frm, f"MITIGATED: TTA (score {threat_score:.2f})", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
                else:
                    # No robust model: use TTA / preprocessing only
                    for cam_pos, pf in pre_frames.items():
                        if pf is None:
                            replaced_camera_dets[cam_pos] = []
                            continue
                        tta_dets = test_time_augment_ensemble(fast_model, pf, augment_fns=None, conf=args.conf)
                        replaced_camera_dets[cam_pos] = tta_dets
                    camera_detections = replaced_camera_dets
                    consensus_list, validated = consensus_system.process_detections(camera_detections, image_sizes)
                    for cam_pos, frm in frames.items():
                        if frm is None: continue
                        cv2.putText(frm, f"MITIGATED: PREPROCESS/TTA (score {threat_score:.2f})", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)

                # draw consensus after mitigation
                for item in consensus_list:
                    cls = item['class_id']; conf = item['final_confidence']; unc = item['uncertainty']
                    for cam_id in range(num_cams):
                        frm = frames.get(cam_id)
                        if frm is None: continue
                        txt = f"AFTER C:{class_list[cls]} Conf:{conf:.2f} Unc:{unc:.2f}"
                        cv2.putText(frm, txt, (10, 50 + 20 * cam_id), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)

            # validated overlays
            for item in validated:
                for cam_id in range(num_cams):
                    frm = frames.get(cam_id)
                    if frm is None: continue
                    cv2.putText(frm, "VALIDATED", (10, 80 + 20 * cam_id), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)

            # show frames
            if DISPLAY_WINDOWS:
                for cam_pos, frm in frames.items():
                    if frm is None: continue
                    cv2.imshow(f"CAM-{cam_pos}", frm)
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        for c in caps:
            c.release()
        cv2.destroyAllWindows()

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Multi-Camera Consensus Demo")
    parser.add_argument("--fast-model", type=str, default="3339model/weights/best.pt", help="Fast YOLO model path")
    parser.add_argument("--robust-model", type=str, default="", help="Robust YOLO model path (optional)")
    parser.add_argument("--class-list", type=str, default="utils/coco.txt", help="Class list file")
    parser.add_argument("--max-cam-probing", type=int, default=4, help="max camera indices to probe")
    parser.add_argument("--conf", type=float, default=0.45, help="detection confidence threshold")
    parser.add_argument("--video-paths", nargs="*", help="optional video files instead of live cameras")
    parser.add_argument("--augment-inference", action="store_true", help="apply light augmentation before inference")
    args = parser.parse_args()
    main(args)
    
