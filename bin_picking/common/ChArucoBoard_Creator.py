import cv2
import numpy as np

# ============================================================
# 1. ArUco Dictionary auswählen
# ============================================================

aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

# ============================================================
# 2. ChArUco Board definieren
# ============================================================

squares_x = 5        # Anzahl Felder in X-Richtung
squares_y = 7        # Anzahl Felder in Y-Richtung
square_length = 0.03 # 4 cm
marker_length = square_length * 0.7     # 0.022 # 3 cm (muss kleiner als square sein)

board = cv2.aruco.CharucoBoard(
    (squares_x, squares_y),
    square_length,
    marker_length,
    aruco_dict
)

# ============================================================
# 3. Board als Bild rendern
# ============================================================

image_size = (2000, 2800)  # Pixelgröße des Outputs

board_image = board.generateImage(image_size)

# ============================================================
# 4. Speichern
# ============================================================

cv2.imwrite("charuco_board.png", board_image)

# optional anzeigen
cv2.imshow("ChArUco Board", board_image)
cv2.waitKey(0)
cv2.destroyAllWindows()