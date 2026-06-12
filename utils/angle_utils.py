import math


def calculate_angle(point_a, point_b, point_c):
    ax, ay = point_a
    bx, by = point_b
    cx, cy = point_c

    ab = (ax - bx, ay - by)
    cb = (cx - bx, cy - by)

    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    mag_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2)

    if mag_ab == 0 or mag_cb == 0:
        return 0

    cos_angle = dot / (mag_ab * mag_cb)
    cos_angle = max(min(cos_angle, 1), -1)

    return round(math.degrees(math.acos(cos_angle)), 2)


def get_landmark_point(landmarks, landmark_id, min_visibility=0.6):
    for landmark in landmarks:
        if landmark["id"] == landmark_id:
            if landmark["visibility"] < min_visibility:
                return None
            return landmark["x"], landmark["y"]

    return None


def get_landmark_vis(landmarks, landmark_id):
    """Return ((x, y), visibility) for a landmark, or (None, 0.0) if absent."""
    for landmark in landmarks:
        if landmark["id"] == landmark_id:
            return (landmark["x"], landmark["y"]), landmark["visibility"]
    return None, 0.0


def extract_squat_features_sideview(landmarks, min_visibility=0.2):
    """
    Side-view squat features computed from the camera-facing leg only.

    In a profile view MediaPipe can see one side of the body reliably and only
    *estimates* the occluded far side, so bilateral features (and especially the
    far-side ankle/foot) are noise. Here we pick whichever side has the higher
    landmark visibility and derive single-leg angles from it. Shoulder/hip are
    near-frontal and stay reliable, so spine lean uses the chosen side too.

    All features are angles -> resolution-invariant and view-consistent.
    """
    ids = {
        "L": {"sho": 11, "hip": 23, "knee": 25, "ankle": 27, "foot": 31},
        "R": {"sho": 12, "hip": 24, "knee": 26, "ankle": 28, "foot": 32},
    }

    pts = {}
    score = {"L": 0.0, "R": 0.0}
    for side, joint_ids in ids.items():
        for joint, lid in joint_ids.items():
            pt, vis = get_landmark_vis(landmarks, lid)
            pts[(side, joint)] = pt
            score[side] += vis

    side = "L" if score["L"] >= score["R"] else "R"

    needed = [pts[(side, j)] for j in ("sho", "hip", "knee", "ankle", "foot")]
    if any(p is None for p in needed):
        return None

    sho, hip, knee, ankle, foot = needed

    knee_angle = calculate_angle(hip, knee, ankle)
    hip_angle = calculate_angle(sho, hip, knee)
    ankle_angle = calculate_angle(knee, ankle, foot)
    spine_angle = vertical_angle(hip, sho)

    return {
        "knee_angle": knee_angle,
        "hip_angle": hip_angle,
        "ankle_angle": ankle_angle,
        "spine_angle": spine_angle,
    }


def vertical_angle(point_a, point_b):
    """
    Angle of line AB relative to vertical axis.
    Used for spine/torso lean estimation.
    """
    ax, ay = point_a
    bx, by = point_b

    dx = bx - ax
    dy = by - ay

    if dy == 0:
        return 90

    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    return round(angle, 2)


def extract_squat_features(landmarks, min_visibility=0.6):
    # min_visibility: lower it (e.g. 0.3) for side-view input, where the
    # far-side limbs are partly occluded and report reduced visibility.
    # MediaPipe landmark IDs
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

    left_shoulder = get_landmark_point(landmarks, LEFT_SHOULDER, min_visibility)
    right_shoulder = get_landmark_point(landmarks, RIGHT_SHOULDER, min_visibility)

    left_hip = get_landmark_point(landmarks, LEFT_HIP, min_visibility)
    right_hip = get_landmark_point(landmarks, RIGHT_HIP, min_visibility)

    left_knee = get_landmark_point(landmarks, LEFT_KNEE, min_visibility)
    right_knee = get_landmark_point(landmarks, RIGHT_KNEE, min_visibility)

    left_ankle = get_landmark_point(landmarks, LEFT_ANKLE, min_visibility)
    right_ankle = get_landmark_point(landmarks, RIGHT_ANKLE, min_visibility)

    left_foot = get_landmark_point(landmarks, LEFT_FOOT_INDEX, min_visibility)
    right_foot = get_landmark_point(landmarks, RIGHT_FOOT_INDEX, min_visibility)

    required = [
        left_shoulder,
        right_shoulder,
        left_hip,
        right_hip,
        left_knee,
        right_knee,
        left_ankle,
        right_ankle,
        left_foot,
        right_foot,
    ]

    if any(point is None for point in required):
        return None

    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

    left_ankle_angle = calculate_angle(left_knee, left_ankle, left_foot)
    right_ankle_angle = calculate_angle(right_knee, right_ankle, right_foot)

    mid_shoulder = (
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2,
    )

    mid_hip = (
        (left_hip[0] + right_hip[0]) / 2,
        (left_hip[1] + right_hip[1]) / 2,
    )

    spine_angle = vertical_angle(mid_hip, mid_shoulder)
    torso_lean = spine_angle

    body_width = abs(left_hip[0] - right_hip[0])
    if body_width == 0:
        body_width = 1

    left_knee_lateral = round(abs(left_knee[0] - left_ankle[0]) / body_width, 2)
    right_knee_lateral = round(abs(right_knee[0] - right_ankle[0]) / body_width, 2)

    symmetry_score = round(
        abs(left_knee_angle - right_knee_angle)
        + abs(left_hip_angle - right_hip_angle)
        + abs(left_ankle_angle - right_ankle_angle),
        2,
    )

    hip_depth = round(mid_hip[1] / 480, 2)

    return {
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "left_hip_angle": left_hip_angle,
        "right_hip_angle": right_hip_angle,
        "left_ankle_angle": left_ankle_angle,
        "right_ankle_angle": right_ankle_angle,
        "spine_angle": spine_angle,
        "torso_lean": torso_lean,
        "left_knee_lateral": left_knee_lateral,
        "right_knee_lateral": right_knee_lateral,
        "symmetry_score": symmetry_score,
        "hip_depth": hip_depth,
    }