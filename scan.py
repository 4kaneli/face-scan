import cv2
import mediapipe as mp
import numpy as np

# ------------------ Parametri Regolabili ------------------

# Colori
TRACK_COLOR = (255, 0, 0)    # Blu elettrico per il wireframe del volto (B, G, R)
TEXT_COLOR = (0, 255, 255)   # Giallo neon per tutte le scritte

# Contatore Blink (Eyes Blinks)
BLINK_OFFSET_X = 180          # Orizzontale: + valori -> più a destra, - valori -> più a sinistra
BLINK_OFFSET_Y = -50          # Verticale: + valori -> più in basso, - valori -> più in alto

# Visualizzazione Head Roll (linea e scritta)
HEAD_LINE_OFFSET_Y = 190      # Quanto sopra la testa mettere la linea
HEAD_TEXT_OFFSET_Y = 30       # Quanto sopra la linea mettere la scritta
HEAD_LINE_LENGTH = 80         # Meta lunghezza della linea orizzontale

# Visualizzazione apertura bocca
MOUTH_OFFSET_X = -230         # Orizzontale: posizione del centro della bocca
MOUTH_OFFSET_Y = 0            # Verticale: posizione del centro della bocca
MOUTH_CIRCLE_MAX = 50         # Dimensione massima del cerchio che si ingrandisce/sgonfia

# Scritta Mouth Open separata dal cerchio
MOUTH_TEXT_OFFSET_X = -35     # Orizzontale: offset della scritta rispetto al centro del cerchio
MOUTH_TEXT_OFFSET_Y = -30     # Verticale: offset della scritta rispetto al centro del cerchio

# Offset cerchio bocca indipendente dalla scritta
MOUTH_CIRCLE_OFFSET_X = 60     # Orizzontale: spostamento cerchio blu
MOUTH_CIRCLE_OFFSET_Y = 20    # Verticale: spostamento cerchio blu

# Offset scritte occhi
LEFT_EYE_TEXT_OFFSET_X = -20
LEFT_EYE_TEXT_OFFSET_Y = 30
RIGHT_EYE_TEXT_OFFSET_X = -20
RIGHT_EYE_TEXT_OFFSET_Y = 30

# Font per tutte le scritte
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.7
TEXT_THICK = 2

# ------------------ Parametri generali ------------------
CAM_INDEX = 0
FRAME_W = 1280
FRAME_H = 720
EYE_CROP_SIZE = 140
EYE_OPEN_THRESHOLD = 0.28

mp_face_mesh = mp.solutions.face_mesh

# Landmark indices
LEFT_EYE_IDX = [33, 133, 159, 145, 158, 153]
RIGHT_EYE_IDX = [362, 263, 386, 373, 387, 380]
LEFT_IRIS_IDX = list(range(468, 473))
RIGHT_IRIS_IDX = list(range(473, 478))
MOUTH_IDX = [13, 14]

# ------------------ Funzioni ------------------

def normalized_to_pixel(norm_landmark, w, h):
    return int(norm_landmark.x * w), int(norm_landmark.y * h)

def eye_aspect_ratio(landmarks, idxs):
    a = np.array([landmarks[idxs[0]].x, landmarks[idxs[0]].y])
    b = np.array([landmarks[idxs[1]].x, landmarks[idxs[1]].y])
    top = np.array([landmarks[idxs[2]].x, landmarks[idxs[2]].y])
    bottom = np.array([landmarks[idxs[3]].x, landmarks[idxs[3]].y])
    hor = np.linalg.norm(a - b)
    ver = np.linalg.norm(top - bottom)
    return ver / hor if hor != 0 else 0

def median_point(landmarks, idxs):
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in idxs])
    return pts.mean(axis=0)

def draw_face_wireframe(image, face_landmarks, w, h):
    overlay = image.copy()
    for conn in mp_face_mesh.FACEMESH_TESSELATION:
        i1, i2 = conn
        p1 = normalized_to_pixel(face_landmarks[i1], w, h)
        p2 = normalized_to_pixel(face_landmarks[i2], w, h)
        cv2.line(overlay, p1, p2, TRACK_COLOR, 1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

def crop_square(image, center, size):
    x, y = center
    half = size // 2
    h, w = image.shape[:2]
    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(w, x + half)
    y2 = min(h, y + half)
    crop = image[y1:y2, x1:x2].copy()
    return cv2.resize(crop, (size, size)) if crop.shape[0] and crop.shape[1] else np.zeros((size, size, 3), dtype=np.uint8)

def draw_pupil(image, center):
    x, y = center
    cv2.circle(image, (x, y), 6, TRACK_COLOR, -1)

# ------------------ Main ------------------

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    blink_counter = 0
    left_prev, right_prev = 'open', 'open'

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                               min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.flip(frame, 1)
            h, w = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            left_state, right_state = 'N/A', 'N/A'

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                draw_face_wireframe(image, landmarks, w, h)

                # Eye aspect ratio e blink
                ear_left = eye_aspect_ratio(landmarks, LEFT_EYE_IDX)
                ear_right = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX)
                left_state = 'open' if ear_left > EYE_OPEN_THRESHOLD else 'close'
                right_state = 'open' if ear_right > EYE_OPEN_THRESHOLD else 'close'

                if left_prev == 'open' and left_state == 'close': blink_counter += 1
                if right_prev == 'open' and right_state == 'close': blink_counter += 1
                left_prev, right_prev = left_state, right_state

                # Eye tiles
                pad = 12
                tile = EYE_CROP_SIZE
                start_x = w - pad - tile - 70
                start_y = pad

                left_eye_center = median_point(landmarks, LEFT_EYE_IDX[:4])
                right_eye_center = median_point(landmarks, RIGHT_EYE_IDX[:4])
                left_eye_px = (int(left_eye_center[0]*w), int(left_eye_center[1]*h))
                right_eye_px = (int(right_eye_center[0]*w), int(right_eye_center[1]*h))

                left_crop = crop_square(image, left_eye_px, tile)
                right_crop = crop_square(image, right_eye_px, tile)

                # Left Eye
                cv2.rectangle(image, (start_x-2, start_y-2), (start_x+tile+2, start_y+tile+2), TRACK_COLOR, 1)
                image[start_y:start_y+tile, start_x:start_x+tile] = left_crop
                cv2.putText(image, f'Left Eye: {left_state}', 
                            (start_x + LEFT_EYE_TEXT_OFFSET_X, start_y+tile+LEFT_EYE_TEXT_OFFSET_Y), 
                            TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICK)

                # Right Eye
                start_y2 = start_y + tile + 50
                cv2.rectangle(image, (start_x-2, start_y2-2), (start_x+tile+2, start_y2+tile+2), TRACK_COLOR, 1)
                image[start_y2:start_y2+tile, start_x:start_x+tile] = right_crop
                cv2.putText(image, f'Right Eye: {right_state}', 
                            (start_x + RIGHT_EYE_TEXT_OFFSET_X, start_y2+tile+RIGHT_EYE_TEXT_OFFSET_Y), 
                            TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICK)

                # Blink counter
                face_center_x = int(np.mean([landmarks[i].x for i in range(33,133)])*w) + BLINK_OFFSET_X
                face_center_y = int(np.mean([landmarks[i].y for i in range(33,133)])*h) + BLINK_OFFSET_Y
                cv2.putText(image, f'Eyes Blinks: {blink_counter}', 
                            (face_center_x, face_center_y), TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICK)

                # Head roll line
                left_eye_px0 = normalized_to_pixel(landmarks[LEFT_EYE_IDX[0]], w, h)
                right_eye_px0 = normalized_to_pixel(landmarks[RIGHT_EYE_IDX[1]], w, h)
                dx = right_eye_px0[0] - left_eye_px0[0]
                dy = right_eye_px0[1] - left_eye_px0[1]
                line_center_x = int((left_eye_px0[0]+right_eye_px0[0])/2)
                line_center_y = int((left_eye_px0[1]+right_eye_px0[1])/2) - HEAD_LINE_OFFSET_Y
                cv2.line(image, 
                         (line_center_x - HEAD_LINE_LENGTH, line_center_y - int(dy/2)), 
                         (line_center_x + HEAD_LINE_LENGTH, line_center_y + int(dy/2)), 
                         TRACK_COLOR, 3)
                cv2.putText(image, 'HEAD ROLL', (line_center_x - 60, line_center_y - HEAD_TEXT_OFFSET_Y), 
                            TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICK)

                # Mouth open visualization
                top_lip = normalized_to_pixel(landmarks[MOUTH_IDX[0]], w, h)
                bottom_lip = normalized_to_pixel(landmarks[MOUTH_IDX[1]], w, h)
                mouth_dist = np.linalg.norm(np.array(top_lip)-np.array(bottom_lip))
                radius = int((mouth_dist/h)*MOUTH_CIRCLE_MAX*3)

                mouth_center_x = left_eye_px[0] + MOUTH_OFFSET_X
                mouth_center_y = int((top_lip[1]+bottom_lip[1])/2) + MOUTH_OFFSET_Y

                percent = int(min(100, radius/MOUTH_CIRCLE_MAX*100))

                # Scritta sopra il cerchio (resta ferma)
                cv2.putText(image, f'Mouth Open: {percent}%', 
                            (mouth_center_x + MOUTH_TEXT_OFFSET_X, mouth_center_y + MOUTH_TEXT_OFFSET_Y), 
                            TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICK)

                # Cerchio blu indipendente
                cv2.circle(image, 
                           (mouth_center_x + MOUTH_CIRCLE_OFFSET_X, mouth_center_y + MOUTH_CIRCLE_OFFSET_Y), 
                           radius, TRACK_COLOR, 2)

            else:
                cv2.putText(image, "Nessun volto rilevato", (20, 40), TEXT_FONT, 1, (0,0,255),2)

            cv2.imshow('Face Scan Wireframe', image)
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

