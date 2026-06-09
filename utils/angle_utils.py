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


def extract_squat_features(landmarks):
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

    left_shoulder = get_landmark_point(landmarks, LEFT_SHOULDER)
    right_shoulder = get_landmark_point(landmarks, RIGHT_SHOULDER)

    left_hip = get_landmark_point(landmarks, LEFT_HIP)
    right_hip = get_landmark_point(landmarks, RIGHT_HIP)

    left_knee = get_landmark_point(landmarks, LEFT_KNEE)
    right_knee = get_landmark_point(landmarks, RIGHT_KNEE)

    left_ankle = get_landmark_point(landmarks, LEFT_ANKLE)
    right_ankle = get_landmark_point(landmarks, RIGHT_ANKLE)

    left_foot = get_landmark_point(landmarks, LEFT_FOOT_INDEX)
    right_foot = get_landmark_point(landmarks, RIGHT_FOOT_INDEX)

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