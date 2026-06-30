# Funktionsindex — bin_picking

Alle öffentlichen Funktionen, Methoden und Klassen sortiert nach Datei.
Format: `Funktion(Parameter)` — Kurzbeschreibung

---

## `camera/camera_interface.py`

| Funktion | Beschreibung |
|---|---|
| `Camera` | Abstrakte Basisklasse für alle Kamera-Backends. |
| `Camera.get_color()` | Liefert einen oder mehrere BGR-Farbframes. |
| `Camera.get_depth()` | Liefert einen oder mehrere Rohtiefenframes (uint16). |

---

## `camera/camera.py`

| Funktion | Beschreibung |
|---|---|
| `RealSenseCamera(serial, adv, align, cfg, buffering)` | Konstruktor — konfiguriert die RealSense-Pipeline und optionales Advanced-Preset. |
| `RealSenseCamera.get_depth(num_frames)` | Nimmt `num_frames` Tiefenbilder auf und gibt sie als Liste zurück. |
| `RealSenseCamera.get_color(num_frames)` | Nimmt `num_frames` Farbbilder auf und gibt sie als Liste zurück. |
| `RealSenseCamera.stream()` | Interaktiver Live-Stream; `s` zum Aufnehmen, `q` zum Beenden. |
| `RealSenseCamera.start()` | Startet die RealSense-Pipeline manuell. |
| `RealSenseCamera.stop()` | Stoppt die RealSense-Pipeline manuell. |
| `RealSenseCamera.start_stream()` | Startet den Hintergrundprozess für Shared-Memory-Pufferung. |
| `RealSenseCamera.stop_stream()` | Stoppt den Hintergrundprozess. |

---

## `common/calibration_errors.py`

| Klasse | Beschreibung |
|---|---|
| `CalibrationError` | Basisklasse für Kalibrierungsfehler. |
| `InvalidRobotTransform` | Wird ausgelöst, wenn der Roboter eine Null- oder None-Pose meldet. |

---

## `common/helper.py`

| Funktion | Beschreibung |
|---|---|
| `get_package_root()` | Gibt den `src/`-Pfad des Workspaces zurück. |
| `get_project_dir()` | Gibt das Stammverzeichnis des `bin_picking`-Pakets zurück. |
| `load_dict_from_json(path)` | Lädt eine JSON-Datei und gibt sie als `dict` zurück. |
| `load_intrinsics(path)` | Lädt Kamera-Intrinsics (`fx`, `fy`, `ppx`, `ppy`, `dist`) aus einer HDF5-Datei. |
| `load_hand_eye(path)` | Lädt die Hand-Eye-Transformation `T_cam2tfp` (4×4) aus einer HDF5-Datei. |
| `write_json_from_dict(data, path)` | Schreibt ein `dict` als JSON-Datei; gibt `True` bei Erfolg zurück. |

---

## `common/image_processing.py`

| Funktion | Beschreibung |
|---|---|
| `ImageProcessing(cfg_path)` | Konstruktor — lädt Filterparameter aus einer JSON-Datei oder nutzt Standardwerte. |
| `ImageProcessing.apply_guided_filter(depth_m, color, cfg)` | Glättet das Tiefenbild per colour-guided Filter unter Erhalt von Tiefenkanten. |
| `ImageProcessing.median_filtering_over_time(img_list)` | Pixelweiser Zeitmedian über eine Liste von Tiefenframes (Nullen werden als ungültig behandelt). |
| `ImageProcessing.generate_point_cloud(depth_m, color, intrinsics, pcd, zmin, zmax)` | Rückprojektion von Tiefen- und Farbbild in eine Open3D-Punktwolke (in-place). |

---

## `common/eye_in_hand_calibration.py`

| Funktion | Beschreibung |
|---|---|
| `get_robot_transform(robot)` | Liest die aktuelle TCP-Pose aus dem Roboter und gibt sie als 4×4-Matrix zurück. |
| `calibrate_camera_charuco(images, ...)` | Kalibriert Kamera-Intrinsics aus ChArUco-Board-Bildern. |
| `get_charuco_pose(color, intrinsics, ...)` | Schätzt die Board-Pose im Kameraframe via PnP; gibt eine 4×4-Matrix oder `None` zurück. |
| `axb_consistency_check(T_gripper2base, T_target2cam, X, logger)` | Berechnet paarweise AX=XB-Residuen für ein Hand-Eye-Ergebnis. |
| `store_calib(rotation, translation, path)` | Speichert die Hand-Eye-Transformation als HDF5-Datei. |
| `undistort_img(img, intrinsics)` | Entfernt Linsenverzeichnung aus einem BGR-Bild. |
| `take_pic(camera, processor, num_depth, num_color)` | Nimmt ein zeitlich gemitteltes Tiefenbild auf. |
| `main(**kwargs)` | Interaktiver Hand-Eye-Kalibrierungsworkflow (22 Posen, ChArUco, `cv.calibrateHandEye`). |

---

## `control/pid.py`

| Funktion | Beschreibung |
|---|---|
| `PID` | Diskreter PID-Regler mit Anti-Windup als Dataclass. |
| `PID.update(actual)` | Aktualisiert den Reglerzustand mit der aktuellen Messung. |
| `PID.apply_control()` | Berechnet und gibt den gesättigten Stellwert [0, 10] V zurück. |
| `PID.set_ref(ref)` | Setzt den Sollwert (Setpoint). |
| `PID.set_control_rate(rate)` | Setzt die Abtastrate in Hz (berechnet dt = 1/rate). |
| `ControlLoop` | Geschlossener Helligkeitsregler für das Ringlicht. |
| `ControlLoop.add_controller(c)` | Bindet eine PID-Instanz als aktiven Regler ein. |
| `ControlLoop.__call__(data)` | Verarbeitet einen BGR-Frame und gibt die neue Steuerspannung zurück. |
| `ControlLoop.auto_update(name, h, w, idx, stop_event, cb, start_event, finish, tol)` | Führt den Regelkreis in einem eigenen Prozess aus; liest Frames aus Shared Memory. |

---

## `pipeline/config.py`

| Funktion / Klasse | Beschreibung |
|---|---|
| `PickResult` | Dataclass mit Greifergebnis: `transform`, `class_name`, `confidence`, `area`. |
| `load_cfg()` | Lädt alle Laufzeit-Konfigurationsdateien aus `data/cfg/` in ein einzelnes `dict`. |

---

## `pipeline/preprocess.py`

| Funktion | Beschreibung |
|---|---|
| `preprocess(color, depth_m, intrinsics, processor)` | Erstellt das 6-Kanal-BGRXYZ-Eingabebild für das YOLO-Modell und die Rohpunktwolke. |
| `unproject(depth, intrinsics)` | Pinhole-Rückprojektion von Tiefen- zu XYZ-Punktwolke (Ausgabe in mm). |
| `build_bgrxyz(color, points)` | Stapelt Farb- und 3D-Kanäle zu einem `(H, W, 6) float32`-Array. |

---

## `pipeline/postprocess.py`

| Funktion | Beschreibung |
|---|---|
| `results_to_masks(results)` | Wandelt ultralytics-`Results`-Objekte in eine flache Liste von Erkennungs-Dicts um. |
| `filter_by_confidence(detections, conf)` | Behält nur Erkennungen mit `confidence >= conf`. |
| `filter_by_percentile(detections, alpha, min_detections)` | Entfernt Erkennungen unterhalb des `alpha`-Quantils der Konfidenzverteilung. |
| `dedup_by_iou(detections, iou_threshold)` | Greedy-NMS-Deduplizierung nach Masken-IoU, klassenweise. |
| `filter_masks(detections, conf, alpha, iou_threshold)` | Wendet die vollständige Filterkette an: Konfidenz → IoU-Dedup → Perzentil. |

---

## `pipeline/inference.py`

| Funktion | Beschreibung |
|---|---|
| `get_model()` | Lädt das YOLO-Segmentierungsmodell aus `data/weights/best.pt` (einmalig gecacht). |
| `run_inference(img6)` | Führt das 6-Kanal-BGRXYZ-Modell auf einem Frame aus; gibt `Results`-Liste zurück. |

---

## `pipeline/verify.py`

| Funktion | Beschreibung |
|---|---|
| `verify_mask(points, mask, class_name, real_areas, tolerance, intrinsics)` | Schätzt die reale Oberfläche einer Maske und vergleicht sie mit dem Referenzwert. |
| `fit_plane(points, sigma_scale)` | Zweistufige SVD-Ebenenschätzung mit Ausreißerfilterung (3-Sigma). |
| `stamp_mask(points, mask)` | Extrahiert gültige 3D-Punkte einer Maske aus der Punktwolke. |
| `fit_plane_and_area(points, sigma_scale)` | Veraltet: SVD-Ebenenfit + Delaunay-Fläche. `verify_mask` mit `intrinsics` bevorzugen. |

---

## `pipeline/occlusion.py`

| Funktion | Beschreibung |
|---|---|
| `fit_circle_3d_icp(points, mask, normal, radius_mm, ...)` | RANSAC + ICP-Kreisfit für teilweise verdeckte Chips; schätzt das wahre Zentrum. |
| `fit_circle_mask(mask)` | Algebraischer Least-Squares-Kreisfit an die 2D-Maskenkontur; gibt `(cx, cy, r)` in Pixeln zurück. |
| `infer_center_3d(cx_px, cy_px, points, mask, normal, intrinsics)` | Rückprojektion eines 2D-Kreiszentrums auf die 3D-Chipebene. |

---

## `pipeline/pose.py`

| Funktion | Beschreibung |
|---|---|
| `calculate_grip_transformation(surface_normal, center, eps)` | Baut eine 4×4-Greiftransformation aus Oberflächennormale und Chipzentrum im Kameraframe. |

---

## `pipeline/grasp.py`

| Funktion | Beschreibung |
|---|---|
| `find_grasp(cam, processor, intrinsics, cfg, robot, hand_eye)` | End-to-End-Greifsuche: Aufnahme → Inferenz → Filter → Verifikation → Erreichbarkeitsprüfung. |

---

## `pipeline/motion.py`

| Funktion | Beschreibung |
|---|---|
| `setup_robot()` | Initialisiert ROS2, erstellt `RobotNode`, wartet auf alle Services; gibt fertigen Knoten zurück. |
| `capture(cam, processor)` | Nimmt ein Farb- + Tiefenbild auf (11 Tiefenframes zeitlich gemittelt, in Meter konvertiert). |
| `move_home(robot, home_pose, tol, timeout)` | Gelenkraum-PTP zur Heimpose; blockiert bis Ankunft oder Timeout. |
| `move_to_grasp(robot, hand_eye, T_camera_target, bin_cfg, tol, timeout)` | Zweiteilige kartesische Annäherung an die Greifpose (Hover → Absenken). |
| `retreat_from_grasp(robot, bin_cfg, drop_pos, tol, timeout)` | Dreistufiger Rückzug nach erfolgreichem Greifen (Heben → Drehen → Fahren). |
| `deliver_chip(robot, class_name, bin_cfg, motion_cfg, tol, timeout)` | Fährt zur klassenspezifischen Ablagepose und schaltet den Sauggreifer aus. |

---

## `pipeline/reachability.py`

| Funktion | Beschreibung |
|---|---|
| `load_chain()` | Lädt die kinematische Kette des KR1205 aus `data/urdf/kr1205.urdf`. |
| `ik_joints(chain, T_world_mm, home_joints_deg)` | Löst IK für eine Zielpose im Weltframe; gibt 7 Gelenkwinkel in Grad zurück oder `None`. |
| `is_reachable(chain, T_world_target_mm, home_joints_deg, bin_cfg)` | Prüft kinematische Erreichbarkeit **und** Kapsel-Kollisionen mit der Behälterwand. |

---

## `pipeline/visualize.py`

| Funktion | Beschreibung |
|---|---|
| `start_visualization(color, points, detections, rejected, result)` | Beendet einen laufenden Visualisierungsprozess und startet einen neuen (nicht blockierend). |
| `stop_visualization()` | Beendet den laufenden Visualisierungsprozess. |
| `visualize_pick(color, points, detections, rejected, result)` | Rendert das Greifergebnis (Masken-Overlay + 3D-Punktwolke); wird im Daemon-Prozess ausgeführt. |

---

## `pipeline/run.py`

| Funktion | Beschreibung |
|---|---|
| `main()` | Einstiegspunkt `ros2 run bin_picking run` — führt einen einzelnen Pick-Zyklus aus und beendet sich. |

---

## `pipeline/loop.py`

| Funktion | Beschreibung |
|---|---|
| `main()` | Einstiegspunkt `ros2 run bin_picking loop` — wiederholt den Pick-Zyklus bis `Ctrl+C` oder leerer Behälter. |
| `_relax_filter(filter_cfg)` | Gibt eine Kopie der Filterparameter mit gelockerten Schwellenwerten zurück (für fast-leere Behälter). |

---

## `robot/node.py`

| Funktion | Beschreibung |
|---|---|
| `RobotNode(name)` | Konstruktor — erstellt alle ROS2-Service-Clients und den State-Topic-Subscriber. |
| `RobotNode.wait_for_service(timeout)` | Blockiert bis alle ROS2-Services verfügbar sind; beendet den Prozess bei Timeout. |
| `RobotNode.spin()` | Startet den ROS2-Executor-Loop (muss in einem Hintergrundthread laufen). |
| `RobotNode.move(config)` | Kartesische Linearbewegung. |
| `RobotNode.move_pnp(pos)` | Gelenkraum-PTP-Bewegung zu 7 Gelenk­winkeln in Grad. |
| `RobotNode.set_sys_frame(pos, rot)` | Definiert den TCP-Frame relativ zum TFC-Frame. |
| `RobotNode.select_jf(ref)` | Wählt den Jogging-Frame nach Index aus. |
| `RobotNode.resume_motion()` | Setzt die Roboterbewegung nach Pause oder Halt fort. |
| `RobotNode.turn_digital_output(index, value)` | Schaltet einen digitalen Ausgang (z. B. Index 6 = Saugventil). |
| `RobotNode.apply_voltage(index, voltage)` | Setzt eine Analogspannung an einem Ausgangskanal. |
| `RobotNode.jog_linear(vel, rot)` | Sendet einen kartesischen Jog-Befehl (nicht blockierend, keine Rückmeldung). |
| `RobotNode.sys_frame(target, ref)` | Fragt die Transformation zwischen zwei benannten Frames ab. |
| `RobotNode.close()` | Zerstört den ROS2-Knoten. |
