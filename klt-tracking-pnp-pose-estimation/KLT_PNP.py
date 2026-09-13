import cv2
import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# 3D coordinates of the 8 box corners (mm converted to meters)
objPoints = np.array([
    [0,   90,  0 ],  # pt 1
    [125, 90,  0 ],  # pt 2
    [125, 90,  70],  # pt 3
    [0,   90,  70],  # pt 4
    [0,   0,   0 ],  # pt 5
    [125, 0,   0 ],  # pt 6
    [125, 0,   70],  # pt 7
    [0,   0,   70],  # pt 8
], dtype=np.float32).reshape((-1, 1, 3)) * 1e-3

IDX_4PTS = [3, 2, 6, 7]
IDX_6PTS = [3, 2, 6, 7, 0, 1]

edges = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

cameraMatrix = np.asarray([
    [606.209, 0,       320.046],
    [0,       606.719, 238.926],
    [0,       0,       1      ]
], dtype=np.float32)
distCoeffs = np.asarray([[0, 0, 0, 0, 0]], dtype=np.float32)

lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

BACKWARD_THRESH = 1.0  # max KLT round-trip error — lower if too many false positives
REPROJ_THRESH   = 2.0  # individual reprojection error above which a point is an outlier — lower to be stricter
REINIT_THRESH   = 3.0  # reprojection error above which a point is excluded from PnP — lower to exclude sooner
TEXTURE_VAR_MIN = 50.0
TEXTURE_WIN     = 15
MIN_PTS_FOR_PNP = 4

# Internal aliases — keep in sync with REPROJ_THRESH / REINIT_THRESH
EXCLUDE_THRESH     = REINIT_THRESH
REINTEGRATE_THRESH = REPROJ_THRESH

# Debug settings:
# - frame 0 always saved
# - every frame from DEBUG_FROM onwards always saved
# - any frame where at least one point exceeds DEBUG_POINT_THRESH is saved
DEBUG_FROM         = 300
DEBUG_POINT_THRESH = 2.0
DEBUG_DIR_SUFFIX   = "debug_frames"


def draw_box(img, rvec, tvec, color=(0, 0, 255)):
    pts, _ = cv2.projectPoints(objPoints, rvec, tvec, cameraMatrix, distCoeffs)
    pts = pts.reshape(-1, 2).astype(int)
    for i, j in edges:
        cv2.line(img, tuple(pts[i]), tuple(pts[j]), color, 2)
    return img


def compute_eqm(pts2d, pts3d, rvec, tvec):
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, cameraMatrix, distCoeffs)
    diff = pts2d.reshape(-1, 2) - proj.reshape(-1, 2)
    return float(np.mean(np.sum(diff**2, axis=1)))


def reproj_errors_individual(pts2d, pts3d, rvec, tvec):
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, cameraMatrix, distCoeffs)
    diff = pts2d.reshape(-1, 2) - proj.reshape(-1, 2)
    return np.linalg.norm(diff, axis=1)


def klt_forward_backward(gray_prev, gray_curr, pts):
    """
    KLT with forward-backward check — rejects points whose round-trip
    error exceeds BACKWARD_THRESH, catching silent drift missed by status=1.
    """
    pts_next, st_fwd, _ = cv2.calcOpticalFlowPyrLK(gray_prev, gray_curr, pts, None, **lk_params)
    pts_back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(gray_curr, gray_prev, pts_next, None, **lk_params)
    err_fb = np.linalg.norm(pts.reshape(-1, 2) - pts_back.reshape(-1, 2), axis=1)
    mask = (st_fwd.ravel() == 1) & (st_bwd.ravel() == 1) & (err_fb < BACKWARD_THRESH)
    return pts_next, mask, err_fb


def local_texture_variance(gray, pts2d, win=TEXTURE_WIN):
    """
    Local image variance around each point.
    KLT needs image gradients — flat regions produce unreliable tracking.
    """
    H, W = gray.shape
    variances = np.zeros(len(pts2d))
    for i, (x, y) in enumerate(pts2d.astype(int)):
        x0, x1 = max(0, x - win), min(W, x + win)
        y0, y1 = max(0, y - win), min(H, y + win)
        patch = gray[y0:y1, x0:x1]
        if patch.size > 0:
            variances[i] = float(np.var(patch))
    return variances


def project_point(pt3d, rvec, tvec):
    proj, _ = cv2.projectPoints(pt3d.reshape(1, 1, 3), rvec, tvec, cameraMatrix, distCoeffs)
    return proj.reshape(1, 1, 2).astype(np.float32)


def is_in_frame(pt2d, H, W, margin=10):
    x, y = pt2d.ravel()
    return margin < x < W - margin and margin < y < H - margin


def draw_points_annotated(frame, good_curr, labels_pts, klt_mask,
                          texture_ok, reproj_err_all, excluded):
    """
    Draw each point colored by its actual status.

    Color legend:
      GREEN  = tracked correctly, reprojection error OK
      ORANGE = KLT backward check failed (drift detected)
      YELLOW = low texture around point (KLT unreliable)
      RED    = high reprojection error (>= REPROJ_THRESH) OR excluded from PnP
               — this is what caused the EQM to rise
    """
    for i, p in enumerate(good_curr.reshape(-1, 2).astype(int)):
        rp = reproj_err_all[i] if reproj_err_all is not None else np.nan

        # RED = any point with high reprojection error OR excluded
        # We check reprojection error FIRST so drifting points show red
        # BEFORE the exclusion mask removes them from PnP
        if not np.isnan(rp) and rp >= REPROJ_THRESH:
            col = (0, 0, 220)       # red — high reprojection error
        elif excluded[i]:
            col = (0, 0, 180)       # dark red — excluded (reprojection was high previously)
        elif not klt_mask[i]:
            col = (0, 165, 255)     # orange — KLT backward check failed
        elif not texture_ok[i]:
            col = (0, 215, 255)     # yellow — low texture
        else:
            col = (0, 220, 0)       # green — good

        cv2.circle(frame, tuple(p), 7, col, -1)
        cv2.circle(frame, tuple(p), 7, (255, 255, 255), 1)  # white outline for visibility

        # Always show reprojection error value next to the point
        rp_str = f"rp={rp:.2f}" if not np.isnan(rp) else "rp=?"
        label  = f"P{labels_pts[i]} {rp_str}"
        # Put label in red if error is high
        lbl_col = (0, 0, 220) if (not np.isnan(rp) and rp >= REPROJ_THRESH) else col
        cv2.putText(frame, label,
                    (p[0] + 9, p[1] - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, lbl_col, 1)


def save_debug_frame(fpath, frame, frame_idx, labels_pts,
                     err_fb, texture_var, texture_ok,
                     reproj_err_all, excluded, klt_mask, current_eqm, H, W):
    """Save annotated frame with full per-point diagnostic panel at the bottom."""
    dbg = frame  # colored points already on frame

    # Dark semi-transparent panel
    ov = dbg.copy()
    cv2.rectangle(ov, (0, H - 125), (W, H), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.65, dbg, 0.35, 0, dbg)

    # Frame number and global MSE
    cv2.putText(dbg, f"Frame {frame_idx:04d}",
                (10, H - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
    eqm_str = f"MSE = {current_eqm:.4f} px2" if current_eqm is not None else "MSE = N/A"
    eqm_col = (0, 80, 255) if (current_eqm is not None and current_eqm > 1.5) else (0, 255, 255)
    cv2.putText(dbg, eqm_str, (160, H - 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, eqm_col, 1)

    # Per-point diagnostic line: fb / rp / tex / status
    for i, lbl in enumerate(labels_pts):
        fb_v  = f"{err_fb[i]:.2f}"
        rp    = reproj_err_all[i] if reproj_err_all is not None else np.nan
        rp_v  = f"{rp:.2f}" if not np.isnan(rp) else "--"
        tx_v  = f"{texture_var[i]:.0f}"

        if excluded[i]:
            status = "EXCL"
            col = (80, 80, 220)
        elif not klt_mask[i]:
            status = "KLT?"
            col = (0, 130, 255)
        elif not texture_ok[i]:
            status = "TEX?"
            col = (0, 200, 255)
        elif not np.isnan(rp) and rp > EXCLUDE_THRESH:
            status = "OUT"
            col = (0, 80, 255)
        else:
            status = "ok"
            col = (0, 210, 0)

        line = f"P{lbl}: fb={fb_v}  rp={rp_v}  tex={tx_v}  [{status}]"
        x_off = 10 + (i % 3) * 215
        y_off = H - 75 + (i // 3) * 22
        cv2.putText(dbg, line, (x_off, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, col, 1)

    cv2.imwrite(fpath, dbg)


def run_pipeline(video_path, point_indices, label, debug=False, output_dir='.'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(os.fspath(video_path))
    ret, first_frame = cap.read()
    if not ret:
        raise IOError("Cannot open video")

    H, W      = first_frame.shape[:2]
    fps       = cap.get(cv2.CAP_PROP_FPS) or 25
    gray_prev = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    N         = len(point_indices)
    labels_pts = [str(i + 1) for i in point_indices]

    debug_dir = None
    if debug:
        debug_dir = output_dir / f"{DEBUG_DIR_SUFFIX}_{label}"
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [DEBUG] → {debug_dir}/")
        print(f"    frame 0 | frames >= {DEBUG_FROM} | frames with any point rp > {DEBUG_POINT_THRESH}px")

    # --- Q1: manually click the N tracked points on the first frame ---
    print(f"\n[{label}] Click {N} points in order: {labels_pts}")
    print("  Press ENTER to confirm.")
    clicked = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < N:
            clicked.append((x, y))
            tmp = first_frame.copy()
            for k, pt in enumerate(clicked):
                cv2.circle(tmp, pt, 7, (0, 255, 0), -1)
                cv2.circle(tmp, pt, 7, (255, 255, 255), 1)
                cv2.putText(tmp, f"P{labels_pts[k]}", (pt[0]+9, pt[1]-9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Init — click points then press ENTER", tmp)

    cv2.imshow("Init — click points then press ENTER", first_frame)
    cv2.setMouseCallback("Init — click points then press ENTER", on_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(clicked) < 4:
        cap.release()
        raise RuntimeError(f"Not enough points clicked ({len(clicked)}/{N}).")
    clicked = clicked[:N]

    # Save frame 0 with clicked points — for the report (Q1 illustration)
    if debug and debug_dir:
        f0 = first_frame.copy()
        for k, pt in enumerate(clicked):
            cv2.circle(f0, pt, 7, (0, 255, 0), -1)
            cv2.circle(f0, pt, 7, (255, 255, 255), 1)
            cv2.putText(f0, f"P{labels_pts[k]}", (pt[0]+9, pt[1]-9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(f0, f"Frame 0000 — initial selection ({N} points)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, "frame_0000_init.jpg"), f0)
        print(f"  [DEBUG] frame 0 saved.")

    pts_prev    = np.array(clicked, dtype=np.float32).reshape(N, 1, 2)
    obj_tracked = objPoints[point_indices]

    out_name = os.fspath(output_dir / f"augmented_{label}.avi")
    out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'XVID'), fps, (W, H))

    eqm_list             = []
    frame_idx            = 0
    last_rvec, last_tvec = None, None
    excluded             = np.zeros(N, dtype=bool)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_curr   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        good_curr   = pts_prev.copy()
        reproj_err_all = np.full(N, np.nan)  # full array, including excluded points
        current_eqm = None

        # --- Q2: KLT with forward-backward check ---
        pts_next, klt_mask, err_fb = klt_forward_backward(gray_prev, gray_curr, pts_prev)
        good_curr = pts_next.copy()

        texture_var = local_texture_variance(gray_curr, good_curr.reshape(-1, 2))
        texture_ok  = texture_var >= TEXTURE_VAR_MIN
        in_frame    = np.array([is_in_frame(good_curr[i], H, W) for i in range(N)])
        usable      = klt_mask & texture_ok & in_frame & ~excluded

        # --- Q3: solvePnP ---
        if int(usable.sum()) >= MIN_PTS_FOR_PNP:
            g_curr = good_curr[usable]
            g_obj  = obj_tracked[usable]

            use_guess = (last_rvec is not None)
            success, rvec, tvec = cv2.solvePnP(
                g_obj, g_curr,
                cameraMatrix, distCoeffs,
                rvec=last_rvec.copy() if use_guess else None,
                tvec=last_tvec.copy() if use_guess else None,
                useExtrinsicGuess=use_guess,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                # Compute reprojection error for ALL points (including excluded)
                # so we can display why a point was excluded even after the fact
                reproj_err_all = reproj_errors_individual(
                    good_curr.reshape(N, 2), obj_tracked.reshape(N, 3), rvec, tvec)

                # Update exclusion mask
                for i in range(N):
                    if excluded[i]:
                        proj_pos = project_point(obj_tracked[i], rvec, tvec)
                        if is_in_frame(proj_pos, H, W):
                            e_test = float(np.linalg.norm(good_curr[i].ravel() - proj_pos.ravel()))
                            if e_test < REINTEGRATE_THRESH:
                                excluded[i] = False
                                good_curr[i] = proj_pos
                    else:
                        if reproj_err_all[i] > EXCLUDE_THRESH:
                            excluded[i] = True

                # Second solvePnP with updated active set
                active_final = ~excluded & in_frame
                if int(active_final.sum()) >= MIN_PTS_FOR_PNP:
                    success2, rvec2, tvec2 = cv2.solvePnP(
                        obj_tracked[active_final],
                        good_curr[active_final],
                        cameraMatrix, distCoeffs,
                        rvec=rvec.copy(), tvec=tvec.copy(),
                        useExtrinsicGuess=True,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    if success2:
                        rvec, tvec = rvec2, tvec2
                        # Recompute reprojection errors with refined pose
                        reproj_err_all = reproj_errors_individual(
                            good_curr.reshape(N, 2), obj_tracked.reshape(N, 3), rvec, tvec)

                last_rvec, last_tvec = rvec.copy(), tvec.copy()

                # --- Q4: project 3D box ---
                frame = draw_box(frame, rvec, tvec)

                # --- Q5: MSE on active inlier points ---
                active_final = ~excluded & in_frame
                if active_final.sum() >= 4:
                    current_eqm = compute_eqm(
                        good_curr[active_final],
                        obj_tracked[active_final],
                        rvec, tvec)
                    eqm_list.append(current_eqm)
                    n_active = int(active_final.sum())
                    cv2.putText(frame,
                                f"[{label}] f={frame_idx} | active={n_active}/{N} | MSE={current_eqm:.3f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Step 1 — draw colored points on frame (modifies frame in place)
        draw_points_annotated(frame, good_curr, labels_pts, klt_mask,
                              texture_ok, reproj_err_all, excluded)

        # Step 2 — save debug frame AFTER points are drawn, so colors are visible
        if debug and debug_dir:
            any_high_rp = not np.all(np.isnan(reproj_err_all)) and \
                          np.nanmax(reproj_err_all) > DEBUG_POINT_THRESH
            if frame_idx >= DEBUG_FROM or any_high_rp:
                tag = "_HIGH" if (any_high_rp and frame_idx < DEBUG_FROM) else ""
                fpath = os.fspath(debug_dir / f"frame_{frame_idx:04d}{tag}.jpg")
                # Pass frame directly (points already drawn) — save_debug_frame
                # only adds the bottom diagnostic panel on top
                save_debug_frame(
                    fpath, frame.copy(), frame_idx, labels_pts,
                    err_fb, texture_var, texture_ok,
                    reproj_err_all, excluded, klt_mask, current_eqm, H, W)

        out.write(frame)
        cv2.imshow(f"Augmented [{label}]", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        gray_prev = gray_curr.copy()
        pts_prev  = good_curr
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    mean_e = float(np.mean(eqm_list)) if eqm_list else float('nan')
    print(f"  [{label}] {frame_idx} frames | mean MSE = {mean_e:.4f} | → {out_name}")
    if debug and debug_dir:
        saved = len([f for f in os.listdir(debug_dir) if f.endswith('.jpg')])
        print(f"  [{label}] {saved} debug frames → {debug_dir}/")
    return eqm_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KLT + PnP box tracking pipeline")
    parser.add_argument(
        "--video",
        type=Path,
        default=Path(__file__).with_name("box_video_data.avi"),
        help="Input video (default: the bundled box_video_data.avi)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("outputs"),
        help="Directory for augmented videos, debug frames and plots",
    )
    args = parser.parse_args()

    video_path = args.video.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    print("\n>>> Q1-Q5: 4 points")
    eqm_4 = run_pipeline(video_path, IDX_4PTS, label="4pts", debug=False,
                          output_dir=output_dir)

    print("\n>>> Q6: 6 points")
    eqm_6 = run_pipeline(video_path, IDX_6PTS, label="6pts", debug=True,
                          output_dir=output_dir)

    # Q7: comparison plot
    plt.figure(figsize=(10, 5))
    plt.plot(eqm_4, label=f"4 points (4,3,7,8) — mean={np.mean(eqm_4):.3f}", color="blue", lw=1.5)
    plt.plot(eqm_6, label=f"6 points (4,3,7,8,1,2) — mean={np.mean(eqm_6):.3f}", color="red",  lw=1.5)
    plt.xlabel("Frame number")
    plt.ylabel("MSE (px²)")
    plt.title("Mean squared reprojection error — 4 pts vs 6 pts")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "eqm_comparison.png", dpi=150)
    plt.show()
