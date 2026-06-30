# bin_picking — API-Referenz

Alle öffentlichen Klassen, Funktionen und Konstanten, geordnet nach Modul.
Einheiten sind durchgehend **Millimeter und Grad**, sofern nicht anders angegeben.

---

## Inhaltsverzeichnis

- [camera](#camera)
  - [camera\_interface.Camera](#camera_interfacecamera)
  - [camera.RealSenseCamera](#camerarealsensecamera)
- [common](#common)
  - [calibration\_errors](#calibration_errors)
  - [helper](#helper)
  - [image\_processing.ImageProcessing](#image_processingimageprocessing)
  - [eye\_in\_hand\_calibration](#eye_in_hand_calibration)
- [control](#control)
  - [pid.PID](#pidpid)
  - [pid.ControlLoop](#pidcontrolloop)
- [pipeline](#pipeline)
  - [config](#config)
  - [preprocess](#preprocess)
  - [postprocess](#postprocess)
  - [inference](#inference)
  - [verify](#verify)
  - [occlusion](#occlusion)
  - [pose](#pose)
  - [grasp](#grasp)
  - [motion](#motion)
  - [reachability](#reachability)
  - [visualize](#visualize)
  - [run](#run)
  - [loop](#loop)
- [robot](#robot)
  - [node.RobotNode](#noderobotnode)

---

## camera

### `camera_interface.Camera`

```python
class Camera(ABC)
```

Abstrakte Basisklasse für alle Kamera-Backends in der Pipeline.

#### Abstrakte Methoden

---

```python
Camera.get_color() -> list
```

Nimmt einen oder mehrere BGR-Farbframes auf und gibt sie zurück.

**Rückgabe** `list[np.ndarray]` — jedes Element ist ein `H×B×3 uint8`-BGR-Array.

---

```python
Camera.get_depth() -> list
```

Nimmt einen oder mehrere Rohtiefenframes auf und gibt sie zurück.

**Rückgabe** `list[np.ndarray]` — jedes Element ist ein `H×B uint16`-Array in
Tiefensensor-Einheiten (mit `RealSenseCamera.unit` multiplizieren für Meter).

---

### `camera.RealSenseCamera`

```python
class RealSenseCamera(Camera)
```

Schlanker pyrealsense2-Wrapper für die Bin-Picking-Pipeline.

Unterstützt zwei Aufnahmemodi:

| Modus | Beschreibung |
|---|---|
| **Direkt** (Standard) | `get_depth()` / `get_color()` blockieren und liefern Frames auf Abruf. |
| **Gepuffert** (`buffering=True`) | Ein Hintergrundprozess streamt Frames in Shared Memory; neuester Frame über `newest_frame` abrufbar. |

#### Konstruktor

```python
RealSenseCamera(
    serial: str = '',
    adv: str = '',
    align: bool = True,
    cfg: str | None = None,
    buffering: bool = False,
)
```

| Parameter | Typ | Beschreibung |
|---|---|---|
| `serial` | `str` | RealSense-Seriennummer. Leer → erstes verfügbares Gerät. |
| `adv` | `str` | Dateiname eines Advanced-Settings-JSON-Presets in `data/realsense_config/` (nur D400-Serie, z. B. `"high_density.json"`). |
| `align` | `bool` | Tiefenframes auf den Farbframe ausrichten. |
| `cfg` | `str \| None` | Stream-Konfigurations-JSON in `data/realsense_config/`. Fällt auf `setup_cfg.json`, dann auf 640×480 @ 30 fps zurück. |
| `buffering` | `bool` | Shared-Memory-Puffermodus aktivieren. |

**Wirft** `FileNotFoundError`, falls `setup_cfg.json` fehlt.  
**Wirft** `RuntimeError`, falls keine Kamera angeschlossen ist.

#### Methoden

---

```python
RealSenseCamera.get_depth(num_frames: int = 1) -> list[np.ndarray]
```

Nimmt `num_frames` Rohtiefenbilder (uint16, aligned falls `align=True`) auf.  
Nur im direkten Modus verfügbar.

---

```python
RealSenseCamera.get_color(num_frames: int = 1) -> list[np.ndarray]
```

Nimmt `num_frames` BGR-uint8-Farbframes auf.  
Nur im direkten Modus verfügbar.

---

```python
RealSenseCamera.stream() -> np.ndarray
```

Interaktiver Farbstream (`s` zum Aufnehmen, `q` zum Beenden).  
Gibt den zuletzt aufgenommenen `H×B×3 uint8`-Frame zurück.  
Nur im direkten Modus verfügbar.

---

```python
RealSenseCamera.start() -> None
RealSenseCamera.stop() -> None
```

Pipeline manuell starten / stoppen.  
Nur im direkten Modus verfügbar.

---

```python
RealSenseCamera.start_stream() -> None
RealSenseCamera.stop_stream() -> None
```

Hintergrundprozess für gepufferten Modus starten / stoppen.  
Nur im gepufferten Modus verfügbar.

---

#### Eigenschaften

| Eigenschaft | Typ | Beschreibung |
|---|---|---|
| `unit` | `float` | Tiefenskalierungsfaktor (Meter pro Roheinheit). |
| `color_resolution` | `tuple[int, int]` | `[Breite, Höhe]` des Farbstreams. |
| `newest_frame` | `tuple[np.ndarray, np.ndarray]` | Neuestes `(color, depth)`-Paar aus dem Shared-Memory-Puffer. Nur im gepufferten Modus. |
| `shm` | `tuple[str, str]` | Shared-Memory-Namen `(depth_name, color_name)`. Nur im gepufferten Modus. |
| `index` | `mp.Value` | Frame-Zähler für den Ringpuffer. Nur im gepufferten Modus. |

---

## common

### `calibration_errors`

```python
class CalibrationError(Exception)
```
Basisklasse für kalibrierungsbezogene Fehler.

---

```python
class InvalidRobotTransform(CalibrationError)
```
Wird ausgelöst, wenn der Roboter eine `None`- oder Nullpose meldet — ein Hinweis
auf einen Kommunikationsfehler oder dass der Roboter nicht referenziert ist.

---

### `helper`

```python
get_package_root() -> Path
```
Geht die Verzeichnishierarchie aufwärts, bis ein Verzeichnis mit einem `src/`-Unterordner
gefunden wird. Gibt den `src/`-Pfad zurück.

---

```python
get_project_dir() -> Path
```
Gibt das Stammverzeichnis des `bin_picking`-Pakets innerhalb des Workspace-`src/` zurück.  
Entspricht `get_package_root() / 'bin_picking'`.

---

```python
load_dict_from_json(path: Path) -> dict
```
Lädt eine JSON-Datei und gibt sie als Python-`dict` zurück.

---

```python
load_intrinsics(path: Path) -> dict
```
Lädt Kamera-Intrinsics aus einer HDF5-Datei.

**Erwartetes HDF5-Layout:**
```
intrinsics/
    mtx   -- 3×3-Kameramatrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    dist  -- Verzerrungskoeffizienten, Form (1,5) oder (5,)
```

**Rückgabe** `dict` mit den Schlüsseln:

| Schlüssel | Typ | Beschreibung |
|---|---|---|
| `fx` | `float` | Brennweite x (Pixel). |
| `fy` | `float` | Brennweite y (Pixel). |
| `ppx` | `float` | Hauptpunkt x (Pixel). |
| `ppy` | `float` | Hauptpunkt y (Pixel). |
| `dist` | `np.ndarray` | Verzerrungskoeffizienten `(k1, k2, p1, p2[, k3, ...])`. |

---

```python
load_hand_eye(path: Path) -> np.ndarray
```
Lädt die Kamera-zu-TFC-Hand-Eye-Transformation `T_cam2tfp` (4×4) aus einer HDF5-Datei.

---

```python
write_json_from_dict(data: dict, path: Path) -> bool
```
Serialisiert `data` als JSON und schreibt es nach `path`. Gibt `True` bei Erfolg zurück.

---

### `image_processing.ImageProcessing`

```python
class ImageProcessing
```

Stellt geführte Tiefenglättung, zeitliches Median-Filtering und
Open3D-Punktwolkengenerierung bereit.

#### Konstruktor

```python
ImageProcessing(cfg_path: str | None = None)
```

| Parameter | Typ | Beschreibung |
|---|---|---|
| `cfg_path` | `str \| None` | Pfad zu einer JSON-Konfigurationsdatei. Falls `None` oder Schlüssel fehlen, werden Standardwerte verwendet. |

**Konfigurationsschlüssel** (mit Standardwerten und gültigen Bereichen):

| Schlüssel | Standard | Bereich | Beschreibung |
|---|---|---|---|
| `guided_radius` | `5` | `[1, 15]` | Nachbarschaftsradius des Guided-Filters (Pixel). |
| `guided_eps` | `1e-3` | `[1e-4, 0.2]` | Regularisierung des Guided-Filters. Kleiner → schärfere Kantentreue. |
| `sharp_lambda` | `0.5` | `[0.0, 1.0]` | Gewichtung der Residual-Schärfung. |
| `residual_clamp` | `0.01` | `[0.01, 0.5]` | Maximales erlaubtes Residual (Meter). |

Werte außerhalb des Bereichs werden mit einer Warnung angepasst, nicht abgelehnt.

#### Methoden

---

```python
ImageProcessing.apply_guided_filter(
    depth_m: np.ndarray,
    color: np.ndarray,
    cfg: dict | None = None,
) -> np.ndarray
```

Wendet einen farbgeführten Filter an, um das Tiefenbild zu glätten und dabei Tiefenkanten zu erhalten.

**Algorithmus:**
1. Sobel-Kantenstärke aus dem Farbbild berechnen.
2. Kantenkarte Gaußsch glätten (Guide).
3. `cv.ximgproc.guidedFilter` anwenden.
4. Gewichtetes, geclamptes Residual addieren (schärft Tiefensprünge ohne Rauschen in flachen Bereichen).

| Parameter | Typ | Beschreibung |
|---|---|---|
| `depth_m` | `np.ndarray` | `H×B float32`-Tiefenbild in **Meter**. |
| `color` | `np.ndarray` | `H×B×3 uint8`-BGR-Farbbild. |
| `cfg` | `dict \| None` | Optionaler Aufruf-Override mit `sobel_ksize` (ungerade `int`) und `gaussian_ksize` (Tupel zweier ungerader `int`s). |

**Rückgabe** `H×B float32`-Tiefenbild in Meter (gleiche Form und Typ wie Eingabe).

**Wirft** `TypeError`, `ValueError`, `KeyError` bei ungültigen Eingaben.

---

```python
ImageProcessing.median_filtering_over_time(
    img_list: list | np.ndarray,
) -> np.ndarray
```

Pixelweiser zeitlicher Median über einen Frame-Stapel. Nullwerte werden als ungültig
(NaN vor dem Median) behandelt, damit sie das Ergebnis nicht verfälschen.

| Parameter | Typ | Beschreibung |
|---|---|---|
| `img_list` | `list \| np.ndarray` | Folge von 2D-Frames gleicher Form. |

**Rückgabe** `H×B float32`-Median-gefiltertes Bild.

**Wirft** `TypeError` falls `img_list` keine Liste oder kein Array ist.  
**Wirft** `ValueError` falls `img_list` leer ist.

---

```python
ImageProcessing.generate_point_cloud(
    depth_m: np.ndarray,
    color: np.ndarray,
    intrinsics: dict,
    pcd: o3d.geometry.PointCloud,
    zmin: float = 0.3,
    zmax: float = 3.0,
) -> None
```

Rückprojiziert ein Tiefen- und Farbbild in eine Open3D-`PointCloud` (**in-place**).
Punkte außerhalb `[zmin, zmax]` Meter werden verworfen.

| Parameter | Typ | Beschreibung |
|---|---|---|
| `depth_m` | `np.ndarray` | `H×B float32`-Tiefenbild in Meter. |
| `color` | `np.ndarray` | `H×B×3 uint8`-BGR-Bild. |
| `intrinsics` | `dict` | Kamera-Intrinsics mit Schlüsseln `fx`, `fy`, `ppx`, `ppy`. |
| `pcd` | `o3d.geometry.PointCloud` | Wird in-place verändert. |
| `zmin` | `float` | Minimale Tiefe (Meter). |
| `zmax` | `float` | Maximale Tiefe (Meter). |

**Wirft** `TypeError`, `ValueError`, `KeyError` bei ungültigen Eingaben.

---

### `eye_in_hand_calibration`

#### Laufzeit-Hilfsfunktion (wird in jedem Pick-Zyklus aufgerufen)

```python
get_robot_transform(robot: RobotNode) -> np.ndarray
```

Fragt die aktuelle TCP-Pose (TFC) vom Roboter ab und gibt sie als 4×4-homogene
Transformation im Roboter-Basisframe zurück.

Der Roboter liefert Euler-XYZ-Winkel (Grad) und einen Translationsvektor (mm);
diese werden zu einer `[R | t; 0 0 0 1]`-Matrix zusammengesetzt.

| Parameter | Typ | Beschreibung |
|---|---|---|
| `robot` | `RobotNode` | Verbundener und laufender Roboterknoten. |

**Rückgabe** `4×4 float64 np.ndarray` — Welt-zu-TFC-Transformation (mm).

**Wirft** `InvalidRobotTransform` falls der Roboter `None` oder eine Nullpose zurückgibt.

---

#### Kalibrierungsfunktionen

```python
main(**kwargs) -> None
```

Interaktiver Hand-Eye-Kalibrierungsworkflow. Fährt den Roboter durch 22 Posen,
nimmt bei jeder Pose Farb- und Tiefenbilder auf, schätzt ChArUco-Posen parallel
und ruft `cv.calibrateHandEye` auf. Speichert das Ergebnis in `data/calibration/hand_eye.hdf5`.

CLI-Flags (via `argparse`):
- `--skip` — Roboterbewegung überspringen, aus `data/buf.hdf5` neu laden.
- `--cam`  — Vorhandene `intrinsics.hdf5` nutzen statt neu zu kalibrieren.

---

```python
calibrate_camera_charuco(
    images: list[np.ndarray],
    board_size: tuple[int, int] = (5, 7),
    square_length_mm: float = 29.7,
    marker_length_mm: float = 22.0,
    aruco_dict_type = aruco.DICT_4X4_250,
) -> dict
```

Kalibriert Kamera-Intrinsics aus einer Liste von ChArUco-Board-Bildern.

**Rückgabe** dict mit den Schlüsseln `fx`, `fy`, `ppx`, `ppy`, `dist`, `rms`.

**Wirft** `ValueError` falls weniger als 4 Bilder verwendbare Ecken liefern.

---

```python
get_charuco_pose(
    color: np.ndarray,
    intrinsics: dict,
    board_size: tuple = (5, 7),
    square_length: float = 29.7,
    marker_length: float = 22.0,
    aruco_dict_id = aruco.DICT_4X4_250,
) -> np.ndarray | None
```

Schätzt die ChArUco-Board-Pose im Kameraframe via PnP.

**Rückgabe** `4×4 float64`-Transformation `T_camera_target` oder `None` bei Detektionsfehler.

---

```python
axb_consistency_check(
    T_gripper2base: list[np.ndarray],
    T_target2cam: list[np.ndarray],
    X: np.ndarray,
    logger: logging.Logger,
) -> tuple[float, float, np.ndarray]
```

Berechnet paarweise `AX = XB`-Residuen für ein Hand-Eye-Kalibrierergebnis.
Große Residuen (> ~5 mm) weisen auf Einheitenfehler, falsche Euler-Konventionen
oder asynchrone Daten hin.

**Rückgabe** `(mean_err, max_err, all_errs)` — translatorische Residuen in der
gleichen Einheit wie die Eingabetransformationen.

---

```python
store_calib(rotation: np.ndarray, translation: np.ndarray, path: Path) -> None
```

Schreibt eine 4×4-Hand-Eye-Transformation als Datensatz `T_cam2tfp` in eine HDF5-Datei.

---

```python
undistort_img(img: np.ndarray, intrinsics: dict) -> np.ndarray
```

Entfernt radiale und tangentiale Linsenverzeichnung aus einem BGR-Bild.

---

```python
take_pic(
    camera: RealSenseCamera,
    processor: ImageProcessing,
    num_depth: int = 9,
    num_color: int = 1,
) -> np.ndarray
```

Nimmt ein zeitlich median-gefiltertes Tiefenbild auf. Gibt ein `H×B float32`-Array zurück.

---

#### Datencontainer

```python
@dataclass
class Data:
    T:     np.ndarray   # 4×4-Roboter-TCP-Transformation im Basisframe (mm)
    depth: np.ndarray   # Zeitlich gefiltertes Tiefenbild
    color: np.ndarray   # BGR-Farbbild
```

---

## control

### `pid.PID`

```python
@dataclass
class PID:
    p: float
    i: float
    d: float
    dt: float = 0.0
    reference: float = 0.0
```

Diskreter PID-Regler mit Anti-Windup.

Der Integralanteil akkumuliert nur, wenn der ungesättigte Ausgang innerhalb der
Hardware-Ausgangsgrenzen `[0, 10]` V liegt.

#### Methoden

---

```python
PID.update(actual: float) -> None
```

Aktualisiert den Reglerzustand mit der aktuellen Messung.

| Parameter | Typ | Beschreibung |
|---|---|---|
| `actual` | `float` | Aktuelle Prozessgröße (z. B. Bildhelligkeit). |

---

```python
PID.apply_control() -> float
```

Berechnet und gibt den gesättigten Stellwert, begrenzt auf `[0, 10]` V, zurück.

---

```python
PID.set_ref(ref: float) -> None
```

Setzt den Sollwert (Setpoint).

---

```python
PID.set_control_rate(rate: float) -> None
```

Setzt die Regelkreisrate in Hz. Berechnet die Abtastperiode `dt = 1 / rate`.

---

#### Eigenschaften

| Eigenschaft | Typ | Beschreibung |
|---|---|---|
| `error` | `float` | Aktueller Regelfehler `Sollwert − Istwert`. |

---

### `pid.ControlLoop`

```python
class ControlLoop
```

Geschlossener Helligkeitsregler für das Ringlicht.

Berechnet die Helligkeit als 67. Perzentil der Graustufenpixelwerte und führt sie
einem `PID`-Regler zu, dessen Ausgabe die Ringlichspannung (0–10 V) steuert.

#### Methoden

---

```python
ControlLoop.add_controller(c: PID) -> None
```

Bindet eine `PID`-Instanz als aktiven Regler ein.

---

```python
ControlLoop.__call__(data: np.ndarray) -> float
```

Verarbeitet einen `H×B×3 uint8`-BGR-Frame und gibt die neue Steuerspannung zurück.

---

```python
ControlLoop.auto_update(
    name: str,
    h: int,
    w: int,
    idx: mp.Value,
    stop_event: mp.Event,
    cb: Callable[[float], None],
    start_event: mp.Event,
    finish: mp.Event,
    tol: float = 0.1,
) -> None
```

Führt den Regelkreis in einem separaten Prozess aus und liest Frames aus Shared Memory.
Gedacht für den Start via `multiprocessing.Process`.

Ruft `cb(voltage)` bei jedem neuen Frame auf. Setzt `finish`, sobald der Ausgang
eingeschwungen ist (Fehler innerhalb `tol` für 5 aufeinanderfolgende Frames oder 2-Sekunden-Timeout).

| Parameter | Typ | Beschreibung |
|---|---|---|
| `name` | `str` | Shared-Memory-Name (aus `RealSenseCamera.shm`). |
| `h`, `w` | `int` | Frame-Höhe und -Breite. |
| `idx` | `mp.Value` | Frame-Zähler (entspricht `RealSenseCamera.index`). |
| `stop_event` | `mp.Event` | Setzen zum Beenden des Loops. |
| `cb` | `Callable` | Wird bei jedem Frame mit der neuen Spannung aufgerufen. |
| `start_event` | `mp.Event` | Muss gesetzt sein, bevor der Regelausgang beginnt. |
| `finish` | `mp.Event` | Wird vom Loop bei Einschwingengesetzt. |
| `tol` | `float` | Helligkeitsfehlertoleraanz zum Eingeschwungen-Erkennen. |

---

## pipeline

### `config`

#### Konstanten

| Name | Wert | Beschreibung |
|---|---|---|
| `TCP_OFFSET` | `np.array([4.587, 0.906, 187.924])` | Werkzeugspitzenversatz vom TFC-Frame in lokalen TFC-Koordinaten (mm). |
| `TCP_TUBE_END_MM` | `197.5` | Saugrohrendabstand vom TFC entlang Werkzeug-Z (mm). |
| `MAX_APPROACH_ANGLE_DEG` | `25.0` | Maximale erlaubte Abweichung der Annäherungsachse von Kamera-+Z (Grad). |

#### `PickResult`

```python
@dataclass
class PickResult:
    transform:  np.ndarray   # 4×4-Greiftransformation im Kameraframe (mm)
    class_name: str           # Erkannte Chip-Klasse (z. B. "Chip_rot")
    confidence: float         # YOLO-Detektionskonfidenz [0, 1]
    area:       float         # Geschätzte reale Oberfläche (mm²)
```

#### Funktionen

---

```python
load_cfg() -> dict
```

Lädt alle Laufzeit-Konfigurationsdateien aus `data/cfg/` in ein einzelnes `dict`.

**Rückgabe:**

| Schlüssel | Quelldatei | Inhalt |
|---|---|---|
| `filter` | `filter_cfg.json` | Detektionsfilterschellenwerte. |
| `real_areas` | `real_areas.json` | Referenzchipflächen pro Klasse (mm²). |
| `home_pose` | `home_pose.json['joint_config']` | 7-elementige Gelenkwinkelliste (Grad). |
| `motion` | `motion_cfg.json` | Ablagepositionen pro Klasse. |
| `bin` | `bin_cfg.json` | Behälterposition im Weltframe und Abmessungen. |
| `chain` | *(berechnet)* | `ikpy.Chain` aus `data/urdf/kr1205.urdf`. |

---

### `preprocess`

```python
preprocess(
    color: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: dict,
    processor: ImageProcessing,
) -> tuple[np.ndarray, np.ndarray]
```

Erstellt das 6-Kanal-BGRXYZ-Eingabebild des Modells und die Rohpunktwolke.

1. Guided Filter auf Tiefe anwenden.
2. Pinhole-Rückprojektion zu 3D (mm).
3. Kanäle auf `[0, 255]` normalisieren (Z logarithmisch).
4. Zu `(H, B, 6) float32`-Array stapeln.

| Parameter | Typ | Beschreibung |
|---|---|---|
| `color` | `np.ndarray` | `H×B×3 uint8`-BGR-Bild. |
| `depth_m` | `np.ndarray` | `H×B float32`-zeitlich gefiltertes Tiefenbild in **Meter**. |
| `intrinsics` | `dict` | Schlüssel: `fx`, `fy`, `ppx`, `ppy`. |
| `processor` | `ImageProcessing` | Stellt den Guided Filter bereit. |

**Rückgabe** `(img6, points)`:
- `img6` — `H×B×6 float32`, Kanäle `[B, G, R, X_norm, Y_norm, Z_norm]` in `[0, 255]`.
- `points` — `H×B×3 float32`, nicht normalisierte Kameraframe-XYZ in **mm**. `NaN` an ungültigen Pixeln.

---

```python
unproject(depth: np.ndarray, intrinsics: dict) -> np.ndarray
```

Standard-Pinhole-Rückprojektion. Eingabetiefe muss in Meter vorliegen; Ausgabe X/Y/Z
in der gleichen Einheit, intern ×1000 → Ausgabe in **mm**.

**Rückgabe** `H×B×3 float32`-Punktwolke. `NaN`, wo Tiefe null oder nicht endlich ist.

---

```python
build_bgrxyz(color: np.ndarray, points: np.ndarray) -> np.ndarray
```

Stapelt Farb- und 3D-Kanäle zu einem `(H, B, 6) float32`-Array skaliert auf `[0, 255]`.
Kanalreihenfolge ist `[B, G, R, X, Y, Z]` — **keine RGB-Konvertierung** (siehe PIPELINE.md §3).

---

### `postprocess`

```python
results_to_masks(results: list[Results]) -> list[dict]
```

Wandelt ultralytics-`Results`-Objekte in eine flache Liste von Detektions-Dicts um.

**Jedes Dict enthält:**

| Schlüssel | Typ | Beschreibung |
|---|---|---|
| `confidence` | `float` | Detektionskonfidenz. |
| `class_id` | `int` | Numerischer Klassenindex. |
| `class_name` | `str` | Lesbarer Klassenname (aus `result.names`). |
| `mask` | `np.ndarray` | `H×B bool`-Segmentierungsmaske. |

---

```python
filter_by_confidence(detections: list[dict], conf: float) -> list[dict]
```

Behält nur Detektionen mit `confidence >= conf`.

---

```python
filter_by_percentile(
    detections: list[dict],
    alpha: float,
    min_detections: int = 5,
) -> list[dict]
```

Entfernt Detektionen unterhalb des `alpha`-Quantils der Konfidenzverteilung.
Wird vollständig übersprungen, wenn `len(detections) <= min_detections`. Der Schwellenwert
wird gelockert, falls striktes Filtern weniger als `min_detections` übrig ließe.

---

```python
dedup_by_iou(detections: list[dict], iou_threshold: float) -> list[dict]
```

Greedy-NMS-Deduplizierung nach Masken-IoU, klassenweise. Detektionen werden in
absteigender Konfidenzreihenfolge verarbeitet; eine Detektion wird unterdrückt,
falls ihr IoU mit einer bereits behaltenen gleichklassigen Detektion `iou_threshold` überschreitet.

---

```python
filter_masks(
    detections: list[dict],
    conf: float,
    alpha: float,
    iou_threshold: float,
) -> list[dict]
```

Wendet die vollständige Filterkette an:
`filter_by_confidence` → `dedup_by_iou` → `filter_by_percentile`.

---

### `inference`

```python
get_model() -> YOLO
```

Lädt das YOLO-Segmentierungsmodell aus `data/weights/best.pt` und cached es für
die Prozesslebensdauer (maximal einmal geladen).

**Wirft** `FileNotFoundError`, falls `best.pt` nicht existiert.

---

```python
run_inference(img6: np.ndarray) -> list[Results]
```

Führt das 6-Kanal-BGRXYZ-Modell auf einem Frame aus.

| Parameter | Typ | Beschreibung |
|---|---|---|
| `img6` | `np.ndarray` | `H×B×6 float32` in `[0, 255]`, wie von `preprocess` erzeugt. **Nicht durch 255 dividieren** — der verwendete ultralytics-Fork normalisiert auf diesem Codepfad nicht. |

**Rückgabe** Liste von ultralytics-`Results`-Objekten (typischerweise ein Element).

---

### `verify`

```python
verify_mask(
    points: np.ndarray,
    mask: np.ndarray,
    class_name: str,
    real_areas: dict,
    tolerance: float,
    intrinsics: dict | None = None,
) -> tuple[bool, float, np.ndarray | None, np.ndarray | None]
```

Schätzt die reale Oberfläche einer Maskenregion und vergleicht sie mit dem
bekannten Referenzwert für `class_name`.

**Flächenmethode:**
- Mit `intrinsics`: pixelweise Methode — `Σ d²/(fx·fy)` über gültige maskierte Pixel,
  dividiert durch `|normal[2]|` zur Neigungskorrektur. Genauer als Delaunay bei nicht-konvexen Formen.
- Ohne `intrinsics`: Delaunay-Triangulation der auf die Ebene projizierten Punktwolke
  (überschätzt bei nicht-konvexen Formen).

**Rückgabe** `(accepted, area_mm2, surface_normal, center)`:

| Element | Typ | Beschreibung |
|---|---|---|
| `accepted` | `bool` | `True` falls `|Fläche − Ref| / Ref ≤ tolerance`. Immer `True`, wenn kein Referenzwert definiert ist. |
| `area_mm2` | `float` | Geschätzte Oberfläche (mm²). |
| `surface_normal` | `np.ndarray \| None` | Einheitsnormalvektor im Kameraframe aus SVD-Ebenenfit. |
| `center` | `np.ndarray \| None` | 3D-Schwerpunkt (mm) im Kameraframe. |

**Wirft** `ValueError`, falls die Maske weniger als 6 gültige 3D-Punkte enthält.

---

```python
fit_plane(points: np.ndarray, sigma_scale: float = 3.0) -> tuple[np.ndarray, np.ndarray]
```

Zweistufiger SVD-Ebenenfit mit Ausreißerelimination (3-Sigma auf Ebenennormaldistanz).

**Rückgabe** `(normal, center)` — beide im gleichen Frame wie `points`.

**Wirft** `ValueError`, falls weniger als 6 Punkte übergeben werden.

---

```python
stamp_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray
```

Extrahiert maskierte Pixel aus einer `H×B×3`-Punktwolke und entfernt NaN-Zeilen.

**Rückgabe** `(N, 3) float32`-Array gültiger 3D-Punkte.

---

```python
fit_plane_and_area(
    points: np.ndarray,
    sigma_scale: float = 3.0,
) -> tuple[float, np.ndarray, np.ndarray]
```

Abwärtskompatible Wrapper-Funktion: SVD-Ebenenfit + Delaunay-Fläche.
**`verify_mask` mit `intrinsics` für die genauere pixelweise Flächenberechnung bevorzugen.**

**Rückgabe** `(area_mm2, normal, center)`.

---

### `occlusion`

```python
fit_circle_3d_icp(
    points: np.ndarray,
    mask: np.ndarray,
    normal: np.ndarray,
    radius_mm: float,
    max_iter: int = 30,
    tol_mm: float = 0.05,
    inlier_thresh_mm: float = 3.0,
) -> np.ndarray
```

Fittet einen Kreis bekannten Radius an die 3D-Randkontur einer teilweise sichtbaren Maske.
Wird als Fallback-Greifzentrumsschätzer verwendet, wenn ein Chip verdeckt ist (Fläche
unter Toleranz) und nur ein Teil seines Umfangs sichtbar ist.

**Algorithmus:**
1. Maskenkonturpixel → 3D-Positionen → auf Chipebene projizieren.
2. RANSAC (fester Radius) für robustes Initial-Zentrum; gerade Schnittkanten an Verdeckungsgrenzen werden verworfen.
3. ICP-Verfeinerung: `c_neu = mean(p_i − r · (p_i − c) / |p_i − c|)`, begrenzt auf die Chipebene.

| Parameter | Typ | Beschreibung |
|---|---|---|
| `points` | `np.ndarray` | `H×B×3 float32`-Kameraframe-Punktwolke (mm). |
| `mask` | `np.ndarray` | `H×B bool`-Detektionsmaske. |
| `normal` | `np.ndarray` | Einheitsoberflächennormale aus `fit_plane`. |
| `radius_mm` | `float` | Bekannter Chipradius `sqrt(ref_area / π)` (mm). |
| `max_iter` | `int` | Maximale ICP-Iterationen. |
| `tol_mm` | `float` | ICP-Konvergenzschwelle (mm). |
| `inlier_thresh_mm` | `float` | RANSAC-Inlier-Bandbreite (mm). |

**Rückgabe** `(3,) float64`-3D-Kreiszentrum auf der Chipebene im Kameraframe (mm).

**Wirft** `ValueError`, falls zu wenige Kontur- oder Chippunkte vorhanden sind oder RANSAC keinen gültigen Kandidaten findet.

---

```python
fit_circle_mask(mask: np.ndarray) -> tuple[float, float, float]
```

Algebraischer Least-Squares-Kreisfit an die 2D-Maskenkontur.

**Rückgabe** `(cx, cy, radius)` in Pixelkoordinaten.

**Wirft** `ValueError`, falls weniger als 5 Konturpunkte gefunden werden oder der Fit einen nicht-positiven Radius liefert.

---

```python
infer_center_3d(
    cx_px: float,
    cy_px: float,
    points: np.ndarray,
    mask: np.ndarray,
    normal: np.ndarray,
    intrinsics: dict,
) -> np.ndarray
```

Rückprojektion eines Pixelraum-Kreiszentrums auf einen 3D-Kameraframe-Punkt.

Nutzt die mediane Maskentiefe als Anfangsschätzung, verfeinert Z dann anhand
der angepassten Ebene, sodass das Ergebnis auf der Chipoberfläche liegt, auch wenn
der Mittelpixel selbst verdeckt ist.

**Rückgabe** `(3,) float64`-3D-Punkt im Kameraframe (mm).

---

### `pose`

```python
calculate_grip_transformation(
    surface_normal: np.ndarray,
    center: np.ndarray,
    eps: float = 5e-5,
) -> np.ndarray
```

Baut eine 4×4-Greiftransformation im Kameraframe aus Oberflächennormale und Mittelpunkt.

Der Greifer nähert sich entlang der Oberflächennormale (Z-Spalte des Ergebnisses);
die In-Ebene-X/Y-Achsen sind beliebig, da der Sauggreifer rotationssymmetrisch ist.

**Konstruktion:**
1. Kreuzprodukt `surface_normal` × `[1, 0, 0]` → `y_axis`.
2. Kreuzprodukt `y_axis` × `surface_normal` → `x_axis`.
3. `surface_normal` umkehren, falls sie nicht zur Kamera zeigt.
4. `y_axis` umkehren, falls Determinante nicht +1 ist (Rechtshändigkeit erzwingen).

| Parameter | Typ | Beschreibung |
|---|---|---|
| `surface_normal` | `np.ndarray` | Einheitsoberflächennormale im Kameraframe. |
| `center` | `np.ndarray` | 3D-Chipmittelpunkt im Kameraframe (mm). |
| `eps` | `float` | Toleranz für die Rechtshändigkeitsprüfung. |

**Rückgabe** `4×4 float32`-homogene Transformation.  
Translationsspalte = `center` (mm); Rotationsspalten = `[x_axis, y_axis, normal]`.

---

### `grasp`

```python
find_grasp(
    cam: RealSenseCamera,
    processor: ImageProcessing,
    intrinsics: dict,
    cfg: dict,
    robot: RobotNode,
    hand_eye: np.ndarray,
) -> PickResult | None
```

End-to-End-Greifsuche: Aufnahme → Inferenz → Filter → Verifikation → Erreichbarkeitsprüfung.

**Rückgabe** das erste gültige `PickResult` (nächster Chip nach medianem Z, der alle
Prüfungen besteht) oder `None`, falls kein greifbarer Chip gefunden wird.

**Pick-Schleife (pro Detektion, sortiert nach aufsteigendem medianem Z):**
1. Maskenflächenprüfung gegen `cfg['real_areas']`.
2. Annäherungswinkel auf `MAX_APPROACH_ANGLE_DEG` begrenzen.
3. IK lösen und Kapselkollisionsprüfung durchführen (`is_reachable`).
4. Falls nicht erreichbar, alternative Annäherungen suchen (`_find_reachable_approach`).
5. Alle Erreichbarkeitsbewertungen werden nach `/tmp/reachability_samples.npy` gespeichert.

**Fallback für verdeckte Chips** (nach der Hauptschleife, falls nichts gefunden):
Für jede wegen zu kleiner Fläche abgelehnte Detektion ICP-Kreisfit (`fit_circle_3d_icp`)
versuchen, um das wahre Zentrum zu ermitteln und erneut zu prüfen.

| Parameter | Typ | Beschreibung |
|---|---|---|
| `cam` | `RealSenseCamera` | Kamerainstanz (direkter Modus). |
| `processor` | `ImageProcessing` | Stellt den Guided Filter bereit. |
| `intrinsics` | `dict` | Kamera-Intrinsics. |
| `cfg` | `dict` | Vollständiges Konfigurations-Dict aus `load_cfg()`. |
| `robot` | `RobotNode` | Laufender Roboterknoten zum Ablesen der aktuellen TCP-Pose. |
| `hand_eye` | `np.ndarray` | `4×4`-Kamera-zu-TFC-Transformation aus `load_hand_eye()`. |

---

### `motion`

```python
setup_robot() -> RobotNode
```

Initialisiert ROS2, erstellt und dreht einen `RobotNode` in einem Hintergrundthread,
wartet auf alle Services und setzt dann den TCP-Frame und den Jogging-Frame.

**Rückgabe** einen einsatzbereiten `RobotNode`.

---

```python
capture(
    cam: RealSenseCamera,
    processor: ImageProcessing,
) -> tuple[np.ndarray, np.ndarray]
```

Nimmt ein Farb- + Tiefenpaar für die Pipeline auf.

- Tiefe: 11 Frames, zeitlich median-gefiltert, auf 100–1000 mm begrenzt, in Meter konvertiert.
- Farbe: 1 Frame.

**Rückgabe** `(color, depth_m)`:
- `color` — `H×B×3 uint8` BGR.
- `depth_m` — `H×B float32` in Meter.

---

```python
move_home(
    robot: RobotNode,
    home_pose: list,
    tol: float = 0.5,
    timeout: float = 30.0,
) -> bool
```

Gelenkraum-PTP-Bewegung zur Heimpose. Blockiert, bis der Roboter innerhalb von
`tol` Grad von `home_pose` angekommen ist oder `timeout` Sekunden vergangen sind.

**Rückgabe** `True` bei Erfolg, `False` bei Timeout oder Roboterhalt/-kollision.

---

```python
move_to_grasp(
    robot: RobotNode,
    hand_eye: np.ndarray,
    T_camera_target: np.ndarray,
    bin_cfg: dict,
    tol: float = 1.0,
    timeout: float = 30.0,
) -> bool
```

Zweistufige kartesische Annäherung an eine Greifpose:
1. **Hover** — auf 180 mm über dem Behälterrand bei Ziel-XY fahren.
2. **Absenken** — senkrecht zur Greif-Z herunterfahren (Zielposition + 6 mm Versatz).

`T_camera_target` liegt im Kameraframe; wird intern über die aktuelle Roboterpose
und die Hand-Eye-Transformation in den Weltframe umgerechnet.

**Rückgabe** `True`, falls beide Bewegungen innerhalb von Toleranz und Timeout abgeschlossen werden.

---

```python
retreat_from_grasp(
    robot: RobotNode,
    bin_cfg: dict,
    drop_pos: list,
    tol: float = 1.0,
    timeout: float = 30.0,
) -> bool
```

Dreistufiger Rückzug nach erfolgreichem Greifen:
1. **Heben** — aufsteigen, bis die Saugrohrespitze den Behälterrand freigibt
   (`bin_top_z + TCP_TUBE_END_MM + 50 mm`).
2. **Drehen** — Werkzeug senkrecht nach unten richten (Welt-−Z), Position beibehalten.
3. **Fahren** — zu `drop_pos` bewegen.

**Rückgabe** `True`, falls alle drei Bewegungen innerhalb von Toleranz und Timeout abgeschlossen werden.

---

```python
deliver_chip(
    robot: RobotNode,
    class_name: str,
    bin_cfg: dict,
    motion_cfg: dict,
    tol: float = 1.0,
    timeout: float = 30.0,
) -> bool
```

Zieht sich zur klassenspezifischen Ablageposition zurück und schaltet dann
Digitalausgang 6 aus (Sauggreifer aus). Fällt auf `motion_cfg['drop_pose']` zurück,
falls keine klassenspezifische Pose für `class_name` definiert ist.

**Rückgabe** `True`, falls die Rückzugsbewegung erfolgreich war.

---

### `reachability`

```python
load_chain() -> ikpy.chain.Chain
```

Lädt die kinematische Kette des Kassow KR1205 aus `data/urdf/kr1205.urdf`.

---

```python
ik_joints(
    chain: Chain,
    T_world_mm: np.ndarray,
    home_joints_deg: list,
) -> list[float] | None
```

Löst IK für ein TCP-Ziel im Weltframe (mm).

Verwendet `home_joints_deg` als Startkonfiguration für den IK-Solver.
Akzeptiert die Lösung nur, wenn der Vorwärtskinematik-Verifikationsfehler
innerhalb von `_IK_TOL_MM = 15 mm` liegt.

**Rückgabe** Liste von 7 Gelenkwinkeln in Grad oder `None`, falls IK nicht konvergiert.

---

```python
is_reachable(
    chain: Chain,
    T_world_target_mm: np.ndarray,
    home_joints_deg: list,
    bin_cfg: dict,
) -> bool
```

Gibt `True` zurück, falls die Greifpose kinematisch erreichbar ist **und** kein
Roboterelement mit dem Behälter kollidiert.

**Kollisionsprüfungen (der Reihe nach):**
1. **Armglieder** — jedes Glied als Kapsel mit konservativem Radius modelliert
   (80 mm an der Schulter, verjüngt auf 40 mm am Handgelenk).
2. **TCP-Baugruppe** — drei Abschnitte entlang Werkzeug-Z:
   - Montageplatte (0–11 mm, r = 120 mm)
   - Übergangsstück (11–27,5 mm, r = 80 mm)
   - Saugrohr (27,5–197,5 mm, r = 12 mm)
3. **Kameragehäuse** — als Kapsel modelliert (r = 7 mm) im Kameraframe,
   über die Hand-Eye-Kalibrierung in den Weltframe transformiert.

Behältergeometrie wird aus `bin_cfg` gelesen (`world_pos`, `size`); nur Wandabschnitte
unterhalb des Behälterrands werden für TCP und Kamera geprüft (der Boden wird immer für Armglieder geprüft).

Beim ersten Ablehnen jedes Kollisionstyps wird ein 3-Ansichten-Diagnosediagramm
unter `plots/reject_<typ>_<timestamp>.png` gespeichert.

---

### `visualize`

```python
start_visualization(
    color: np.ndarray,
    points: np.ndarray,
    detections: list[dict],
    rejected: list[dict],
    result: PickResult,
) -> None
```

Beendet einen laufenden Visualisierungsprozess und startet einen neuen (nicht blockierend).

---

```python
stop_visualization() -> None
```

Beendet den laufenden Visualisierungsprozess.

---

```python
visualize_pick(
    color: np.ndarray,
    points: np.ndarray,
    detections: list[dict],
    rejected: list[dict],
    result: PickResult,
) -> None
```

Rendert das Greifergebnis. Läuft im Daemon-Prozess, der von `start_visualization` gestartet wird.

**Linkes Bild** — Farbbild mit Masken-Overlays:
- Grün: Gewinner-Detektion.
- Orange: akzeptierte Kandidaten, die nicht gewählt wurden.
- Grau + ✕: abgelehnte Detektionen.
- Schwarzer Punkt: nächstliegender Pixel zum 3D-Greifzentrum.

**Rechtes Bild** — 3D-Punktwolken-Scatter mit rotem Pfeil als Annäherungsvektor.

Speichert das Diagramm unter `plots/pick_<unix_timestamp>.png`, zeigt es dann interaktiv an.

---

### `run`

```python
main() -> None
```

**Einstiegspunkt:** `ros2 run bin_picking run`

Initialisiert alle Ressourcen (Kamera, Roboter, Prozessor, Kalibrierdaten, Konfiguration),
führt einen vollständigen Pick-Zyklus durch und fährt dann ROS2 herunter.

Pick-Zyklus:
```
move_home → find_grasp → move_to_grasp → Sauggreifer an (0,35 s) → deliver_chip → move_home
```

Zeitnahme für die Aufnahme-bis-Pose-Phase wird auf `INFO`-Ebene protokolliert.

---

### `loop`

```python
main() -> None
```

**Einstiegspunkt:** `ros2 run bin_picking loop`

Wie `run.main()`, wiederholt den Zyklus aber unbegrenzt bis `Ctrl+C`.

Der Roboter kehrt vor **jeder** Aufnahme zur Heimpose zurück (erforderlich, da die
Kamera Eye-in-Hand montiert ist und vor jeder Aufnahme an der bekannten Beobachtungspose
sein muss).

---

## robot

### `node.RobotNode`

```python
class RobotNode(rclpy.node.Node)
```

ROS2-Knoten, der die Service- und Topic-Schnittstelle des Kassow-Robotercontrollers kapselt.

Alle Bewegungs- und E/A-Aufrufe nutzen intern asynchrone ROS2-Service-Calls, die hier
über `threading.Event`-Blocking als synchron wirkende Methoden bereitgestellt werden.

**Der Knoten muss in einem Hintergrundthread gedreht werden**, damit der ROS2-Executor
Service-Antworten zustellen kann, während Aufrufer blockieren.

```python
rclpy.init()
robot = RobotNode()
threading.Thread(target=robot.spin, daemon=True).start()
robot.wait_for_service()
```

Verwendete ROS2-Topics und -Services:

| Name | Typ | Rolle |
|---|---|---|
| `kr/motion/move_linear` | `MoveLinear` (srv) | Kartesische Linearbewegung. |
| `kr/motion/move_joint` | `MoveJoint` (srv) | Gelenkraum-PTP-Bewegung. |
| `kr/motion/pause` | `PauseMotion` (srv) | Bewegung pausieren. |
| `kr/motion/resume` | `ResumeMotion` (srv) | Bewegung fortsetzen. |
| `kr/motion/select_jogging_frame` | `SelectJoggingFrame` (srv) | Jogging-Frame wählen. |
| `kr/robot/get_robot_pose` | `GetRobotPose` (srv) | Kartesische TCP-Pose abfragen. |
| `kr/robot/get_system_frame` | `GetSystemFrame` (srv) | Benannte Frame-Transformation abfragen. |
| `kr/robot/set_system_frame` | `SetSystemFrame` (srv) | Benannten Frame definieren (TCP-Versatz). |
| `kr/system/get_robot_state` | `GetRobotState` (srv) | Roboterstatus abfragen. |
| `kr/system/state` | `SystemState` (topic) | Roboterstatus + Drehmoment + Gelenkpositionsstrom. |
| `kr/motion/jog_linear` | `JogLinear` (topic) | Kontinuierliches kartesisches Joggen. |
| `kr/iob/set_digital_output` | `SetDiscreteOutput` (srv) | Digitalausgang setzen (Sauggreifer). |
| `kr/iob/set_voltage_output` | `SetAnalogOutput` (srv) | Analogspannung setzen (Ringlicht). |

#### Konstruktor

```python
RobotNode(name: str = 'robot_node')
```

#### Methoden

---

```python
RobotNode.wait_for_service(timeout: float = 5.0) -> None
```

Blockiert, bis alle ROS2-Service-Clients verfügbar sind. Beendet den Prozess, falls
ein Service innerhalb von `timeout` Sekunden nicht erreichbar ist.

---

```python
RobotNode.spin() -> None
```

Startet den ROS2-Executor-Loop. Muss in einem Hintergrundthread aufgerufen werden.

---

```python
RobotNode.move(config: dict) -> bool
```

Kartesische Linearbewegung.

---

```python
RobotNode.move_pnp(pos: list[float]) -> bool
```

Gelenkraum-PTP-Bewegung.

| Parameter | Typ | Beschreibung |
|---|---|---|
| `pos` | `list[float]` | 7 Gelenkwinkel in Grad. |

**Rückgabe** `True`, falls die Bewegungsanfrage akzeptiert wurde.

---

```python
RobotNode.set_sys_frame(pos: list, rot: list) -> None
```

Definiert den TCP-Frame relativ zum TFC.

| Parameter | Typ | Beschreibung |
|---|---|---|
| `pos` | `list` | `[x, y, z]`-Versatz (mm). |
| `rot` | `list` | `[rx, ry, rz]` XYZ-Euler-Winkel (Grad). |

---

```python
RobotNode.select_jf(ref: int = 2) -> None
```

Wählt den Jogging-Frame nach Index aus.

---

```python
RobotNode.resume_motion() -> None
```

Setzt die Roboterbewegung nach Pause oder Halt fort.

---

```python
RobotNode.turn_digital_output(index: int, value: int) -> None
```

Setzt einen Digitalausgang (Fire-and-Forget).

| Parameter | Typ | Beschreibung |
|---|---|---|
| `index` | `int` | Ausgangsindex (6 = Saugventil). |
| `value` | `int` | `1` zum Aktivieren, `0` zum Deaktivieren. |

---

```python
RobotNode.apply_voltage(index: int, voltage: float) -> None
```

Setzt eine Analogausgangsspannung (Fire-and-Forget).

| Parameter | Typ | Beschreibung |
|---|---|---|
| `index` | `int` | Analogkanalindex. |
| `voltage` | `float` | Zielspannung (V). |

---

```python
RobotNode.jog_linear(vel: list, rot: list) -> None
```

Sendet einen kartesischen Jog-Befehl (nicht blockierend, keine Rückmeldung).

| Parameter | Typ | Beschreibung |
|---|---|---|
| `vel` | `list[float]` | `[vx, vy, vz]` Lineargeschwindigkeit. |
| `rot` | `list[float]` | `[wx, wy, wz]` Winkelgeschwindigkeit. |

---

```python
RobotNode.sys_frame(target: str, ref: str) -> np.ndarray | None
```

Fragt die Transformation von Frame `ref` zu Frame `target` ab.

**Rückgabe** `2×3 float64`-Array `[[px, py, pz], [rx, ry, rz]]` (mm und Grad),
oder `None` bei Fehler.

---

```python
RobotNode.close() -> None
```

Zerstört den ROS2-Knoten.

---

#### Eigenschaften

| Eigenschaft | Typ | Beschreibung |
|---|---|---|
| `robot_pose` | `dict` | Aktuelle kartesische TCP-Pose: `{'pos': [x,y,z], 'rot': [rx,ry,rz]}`. |
| `pose` | `any` | Neuester Gelenkpositionseintrag aus dem SystemState-Topic. |
| `sys_state_full` | `tuple[np.ndarray, np.ndarray]` | Vollständige Historie der `(states, torques)`-Arrays seit Knotenstart. |
| `readable` | `bool` | Vom SystemState-Callback auf `True` gesetzt, wenn Roboterstatus == 6 (Halt/Kollision). Wird von `_poll_until_close` nach Behandlung zurückgesetzt. |
