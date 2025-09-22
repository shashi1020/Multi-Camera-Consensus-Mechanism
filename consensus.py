import numpy as np
import cv2
from ultralytics import YOLO
import random
from collections import deque, defaultdict
import time

# -------------------------
# Utilities (your augment + predictions slightly adapted)
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
    """Draw YOLO detections (list of dicts with bbox, class_id, confidence)."""
    for det in detections:
        bb = det['bbox']  # [x1, y1, x2, y2] (float)
        clsID = det['class_id']
        conf = det['confidence']
        cv2.rectangle(
            img,
            (int(bb[0]), int(bb[1])),
            (int(bb[2]), int(bb[3])),
            Box_colours[int(clsID)],
            2,
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            img,
            f"{class_list[int(clsID)]} {conf:.2f}",
            (int(bb[0]), int(bb[1]) - 6),
            font,
            0.6,
            (255, 0, 255),
            2,
        )

# -------------------------
# Consensus class
# -------------------------
class IntelligentMultiCameraConsensus:
    def __init__(
        self,
        num_cameras,
        camera_positions=None,
        confidence_threshold=0.5,
        center_distance_threshold=0.12,  # normalized units (0..1)
        temp_scale=1.2,
        history_len=6,
        min_persistence=2,
    ):
        self.num_cameras = num_cameras
        self.confidence_threshold = confidence_threshold
        self.camera_reliability_scores = np.ones(num_cameras, dtype=float)
        self.center_distance_threshold = center_distance_threshold
        self.temp_scale = temp_scale
        self.history = deque(maxlen=history_len)  # store past consensus lists for temporal validation
        self.min_persistence = min_persistence  # how many frames consensus must appear to be validated
        # camera_positions optional; if None we use simple geometric factor = 1/(1+dist)
        self.camera_positions = camera_positions or {i: np.array([i, 0, 0], dtype=float) for i in range(num_cameras)}

    # Main entry
    def process_detections(self, camera_detections, image_sizes):
        """
        camera_detections: dict[camera_id] = list of detections per camera (bbox in xyxy pixel coords, confidence, class_id)
        image_sizes: dict[camera_id] = (width, height)
        returns: list of consensus detections (bbox in pixel coords averaged per cluster per camera), and validated list
        """
        # Step 0: normalize detection centers and sizes for spatial comparison
        normalized = self._normalize_detections(camera_detections, image_sizes)

        # Step 1: calibrate confidences
        self._calibrate_confidences(normalized)

        # Step 2: build clusters across cameras (simple greedy matching by class + center distance)
        clusters = self._cluster_across_cameras(normalized)

        # Step 3: compute spatial correlations (per cluster -> support per member detection)
        self._compute_spatial_support(clusters)

        # Step 4: consensus scoring per cluster
        consensus_list = self._make_consensus(clusters)

        # Step 5: temporal validation
        validated = self._temporal_validation(consensus_list)

        # Step 6: update camera reliability
        self._update_camera_reliability(clusters, consensus_list)

        # store in history
        self.history.append(consensus_list)

        return consensus_list, validated

    # Helpers
    def _normalize_detections(self, camera_detections, image_sizes):
        norm = {}
        for cam_id, dets in camera_detections.items():
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
        # temperature scaling using logit
        def temp_scale(conf):
            # avoid 0/1
            eps = 1e-6
            conf = np.clip(conf, eps, 1 - eps)
            logit = np.log(conf / (1 - conf))
            scaled = 1 / (1 + np.exp(-logit / self.temp_scale))
            return float(scaled)

        for cam_id, dets in normalized.items():
            for d in dets:
                d['confidence_cal'] = temp_scale(d['confidence'])

    def _cluster_across_cameras(self, normalized):
        """
        Create clusters (object hypotheses) by greedy matching:
        - iterate over all detections (any camera),
        - if matches an existing cluster by class & center distance, add there,
        - else create new cluster.
        """
        clusters = []  # each cluster is dict with 'members': list of detection dicts
        for cam_id, dets in normalized.items():
            for d in dets:
                placed = False
                for cl in clusters:
                    # require same class
                    if d['class_id'] != cl['class_id']:
                        continue
                    # distance to cluster centroid
                    centroid = cl['centroid']
                    dist = np.linalg.norm(d['center'] - centroid)
                    if dist <= self.center_distance_threshold:
                        cl['members'].append(d)
                        # update centroid weighted by size (equal weight here)
                        all_centers = np.array([m['center'] for m in cl['members']])
                        cl['centroid'] = np.mean(all_centers, axis=0)
                        placed = True
                        break
                if not placed:
                    clusters.append({
                        'class_id': d['class_id'],
                        'members': [d],
                        'centroid': d['center'].copy()
                    })
        return clusters

    def _compute_spatial_support(self, clusters):
        # For each cluster, compute pairwise spatial correlation between member detections
        for cl in clusters:
            members = cl['members']
            n = len(members)
            if n <= 1:
                # low support
                for m in members:
                    m['spatial_support'] = 0.0
                continue

            # form matrix of distance-based correlations
            supports = []
            for i, mi in enumerate(members):
                # compute support of mi from other members
                s = 0.0
                for j, mj in enumerate(members):
                    if i == j: continue
                    # detection overlap correlation (center distance + size similarity)
                    center_dist = np.linalg.norm(mi['center'] - mj['center'])
                    center_score = np.exp(- (center_dist / (self.center_distance_threshold + 1e-6))**2)
                    size_diff = np.linalg.norm(mi['size'] - mj['size'])
                    size_score = 1.0 / (1.0 + size_diff*10.0)  # heuristic
                    # geometric factor based on camera positions
                    pos_i = self.camera_positions[mi['cam']]
                    pos_j = self.camera_positions[mj['cam']]
                    cam_dist = np.linalg.norm(pos_i - pos_j)
                    geom_factor = 1.0 / (1.0 + cam_dist)
                    corr = center_score * size_score * geom_factor
                    s += corr
                s /= max(1, n-1)
                supports.append(s)
            # assign per member
            for m, s in zip(members, supports):
                m['spatial_support'] = float(s)

    def _make_consensus(self, clusters):
        consensus_list = []
        for cl in clusters:
            members = cl['members']
            # weight each member by calibrated confidence * camera reliability * (1 + spatial_support)
            weights = []
            for m in members:
                cam_rel = float(self.camera_reliability_scores[m['cam']])
                weight = m.get('confidence_cal', m['confidence']) * cam_rel * (1.0 + m.get('spatial_support', 0.0))
                weights.append(weight)
            weights = np.array(weights) + 1e-8
            # final confidence = weighted average of calibr. confidences
            final_conf = float(np.sum([m.get('confidence_cal', m['confidence']) * w for m, w in zip(members, weights)]) / np.sum(weights))
            # uncertainty: inverse of average spatial support (higher support -> lower uncertainty)
            avg_sp = float(np.mean([m.get('spatial_support', 0.0) for m in members]))
            uncertainty = float(1.0 / (1.0 + avg_sp))
            # consensus strength: std of member weights (lower -> stronger consensus)
            consensus_strength = float(np.std(weights))

            # compute a per-camera bbox guess for display: use boxes from each member's 'orig'
            per_camera_bboxes = defaultdict(lambda: None)
            for m in members:
                per_camera_bboxes[m['cam']] = m['orig']['bbox']  # pixel coords

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
        """
        Validate consensus list using short history:
        - If an object (same class & similar centroid) appears in at least min_persistence frames -> validated.
        """
        validated = []
        # Build signature for items in current frame
        current_signatures = []
        for item in consensus_list:
            # compute average centroid in image-normalized coords across members
            centroid = np.mean([m['center'] for m in item['members']], axis=0)
            current_signatures.append((item['class_id'], centroid))

        # count occurrences in history (including current)
        for idx, (cls, cent) in enumerate(current_signatures):
            count = 1  # current
            for past in self.history:
                for p in past:
                    if p['class_id'] != cls:
                        continue
                    # compute centroid for past cluster
                    p_cent = np.mean([m['center'] for m in p['members']], axis=0)
                    if np.linalg.norm(p_cent - cent) <= self.center_distance_threshold:
                        count += 1
                        break
            # validated if count >= min_persistence
            if count >= self.min_persistence:
                validated.append(consensus_list[idx])
        return validated

    def _update_camera_reliability(self, clusters, consensus_list):
        """
        Increase reliability for cameras that participated in validated consensus,
        slightly decrease for those that didn't.
        """
        # baseline decay
        decay = 0.995
        self.camera_reliability_scores *= decay

        # reward cameras that contributed to clusters with decent confidence
        for item in consensus_list:
            score = item['final_confidence'] * (1.0 - item['uncertainty'])
            for m in item['members']:
                cam = m['cam']
                self.camera_reliability_scores[cam] += 0.01 * score

        # keep reliability in [0.2, 2.0] to avoid extremes
        self.camera_reliability_scores = np.clip(self.camera_reliability_scores, 0.2, 2.0)


# -------------------------
# Integrate with your capture loop
# -------------------------
if __name__ == "__main__":
    with open("utils/coco.txt") as f:
        class_list = f.read().strip().split("\n")

    Box_colours = []
    for i in range(len(class_list)):
        R = random.randint(0, 255)
        G = random.randint(0, 255)
        B = random.randint(0, 255)
        Box_colours.append((B, G, R))

    model = YOLO(r"3339model\weights\best.pt", "v8")

    caps = [cv2.VideoCapture(i) for i in range(2)]
    for cap in caps:
        cap.set(3, 640)
        cap.set(4, 480)
        cap.set(10, 100)

    if any(not cap.isOpened() for cap in caps):
        print("Unable to open one or more webcams. Exiting...")
        # exit()
    else:
        print("Starting your webcams...")

    num_cams = len(caps)
    # Example: give approximate camera positions in meters (if unknown, just use different x coords)
    camera_positions = {0: np.array([0.0, 0.0, 0.0]), 1: np.array([1.0, 0.0, 0.0])}
    consensus_system = IntelligentMultiCameraConsensus(
        num_cameras=num_cams,
        camera_positions=camera_positions,
        confidence_threshold=0.5,
        center_distance_threshold=0.12,
        temp_scale=1.2,
        history_len=6,
        min_persistence=2
    )

    # If you want to highlight only 'weapon' classes, set indexes here (COCO uses e.g. 'knife'/'sports ball' etc.)
    # Replace with the class ids matching your model labels for weapon classes
    weapon_class_ids = set([])  # e.g. {41, 42} - fill if you want to only treat some classes as weapons

    while True:
        # Read frames
        frames = {}
        image_sizes = {}
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                frames[i] = None
            else:
                frames[i] = frame
                h, w = frame.shape[:2]
                image_sizes[i] = (w, h)

        if any(fr is None for fr in frames.values()):
            print("Unable to load frames. Exiting...")
            break

        # Perform per-camera detection (on augmented images) and collect info
        camera_detections = {}
        for i, frame in frames.items():
            aug = augment_image(frame)
            Detect_obj = model.track(source=[aug], conf=0.45, save=False)  # lowered conf to gather more votes
            dets = []
            # Access as you previously did; ensure indexing safe
            if len(Detect_obj[0]) > 0:
                boxes = Detect_obj[0].boxes
                for b in boxes:
                    bb = b.xyxy.numpy()[0]          # [x1, y1, x2, y2]
                    conf = float(b.conf.numpy()[0])
                    cls = int(b.cls.numpy()[0])
                    # Only include classes you care about (optional)
                    dets.append({'bbox': bb, 'confidence': conf, 'class_id': cls})
            camera_detections[i] = dets

        # Draw per-camera detector outputs (for visualization)
        for i, frame in frames.items():
            draw_detections(frame, camera_detections[i], Box_colours, class_list)

        # Apply consensus on the collected detections
        consensus_list, validated = consensus_system.process_detections(camera_detections, image_sizes)

        # Visualize consensus results: draw per-camera consensus bboxes and overlay info
        for item in consensus_list:
            cls = item['class_id']
            conf = item['final_confidence']
            unc = item['uncertainty']
            strength = item['consensus_strength']

            # Show a small text overlay on each camera: class, conf, unc
            for cam_id in range(num_cams):
                cam_frame = frames[cam_id]
                txt = f"C:{class_list[cls]} Conf:{conf:.2f} Unc:{unc:.2f}"
                cv2.putText(cam_frame, txt, (10, 30 + 20 * cam_id), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # If consensus provides per-camera bbox for this cam, draw it
                if cam_id in item['per_camera_bboxes'] and item['per_camera_bboxes'][cam_id] is not None:
                    bb = item['per_camera_bboxes'][cam_id]
                    cv2.rectangle(cam_frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 255, 0), 3)

        # Optionally mark validated items differently
        for item in validated:
            for cam_id in range(num_cams):
                cam_frame = frames[cam_id]
                cv2.putText(cam_frame, "VALIDATED", (10, 60 + 20 * cam_id), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # Show frames
        for i, frame in frames.items():
            cv2.imshow(f'CAM-{i}', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
