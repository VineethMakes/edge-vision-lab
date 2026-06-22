use serde::{Deserialize, Serialize};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

const FRAME_WIDTH: usize = 96;
const FRAME_HEIGHT: usize = 54;
const FRAME_PIXELS: usize = FRAME_WIDTH * FRAME_HEIGHT;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub fn center(self) -> Point {
        Point {
            x: self.x + self.width * 0.5,
            y: self.y + self.height * 0.5,
        }
    }

    pub fn contains(self, point: Point) -> bool {
        point.x >= self.x
            && point.x <= self.x + self.width
            && point.y >= self.y
            && point.y <= self.y + self.height
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraCalibration {
    pub frame_width: usize,
    pub frame_height: usize,
    pub focal_length_px: f32,
    pub mounting_height_m: f32,
    pub pitch_deg: f32,
    pub regions: Vec<RegionOfInterest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionOfInterest {
    pub name: String,
    pub bounds: Rect,
    pub sensitivity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub class_name: String,
    pub confidence: f32,
    pub bbox: Rect,
    pub centroid: Point,
    pub motion_score: f32,
    pub region: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Track {
    pub id: u32,
    pub class_name: String,
    pub bbox: Rect,
    pub centroid: Point,
    pub velocity: Point,
    pub age_frames: u32,
    pub missed_frames: u32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub capture_ms: f32,
    pub preprocess_ms: f32,
    pub detection_ms: f32,
    pub tracking_ms: f32,
    pub total_ms: f32,
    pub fps: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineSnapshot {
    pub frame_index: u64,
    pub timestamp_ms: f32,
    pub width: usize,
    pub height: usize,
    pub grayscale: Vec<u8>,
    pub motion_heatmap: Vec<u8>,
    pub detections: Vec<Detection>,
    pub tracks: Vec<Track>,
    pub latency: LatencyStats,
    pub calibration: CameraCalibration,
}

#[derive(Debug, Clone)]
struct SyntheticTarget {
    class_name: &'static str,
    origin: Point,
    radius: Point,
    size: Point,
    phase: f32,
    speed: f32,
    intensity: u8,
}

impl SyntheticTarget {
    fn bbox_at(&self, time: f32) -> Rect {
        let x = self.origin.x + (time * self.speed + self.phase).sin() * self.radius.x;
        let y = self.origin.y + (time * self.speed * 0.71 + self.phase).cos() * self.radius.y;
        Rect {
            x: (x - self.size.x * 0.5).max(1.0),
            y: (y - self.size.y * 0.5).max(1.0),
            width: self.size.x,
            height: self.size.y,
        }
    }
}

impl Default for CameraCalibration {
    fn default() -> Self {
        Self {
            frame_width: FRAME_WIDTH,
            frame_height: FRAME_HEIGHT,
            focal_length_px: 72.0,
            mounting_height_m: 2.6,
            pitch_deg: 18.0,
            regions: vec![
                RegionOfInterest {
                    name: "driveway".to_string(),
                    bounds: Rect {
                        x: 6.0,
                        y: 28.0,
                        width: 38.0,
                        height: 20.0,
                    },
                    sensitivity: 0.66,
                },
                RegionOfInterest {
                    name: "walkway".to_string(),
                    bounds: Rect {
                        x: 45.0,
                        y: 18.0,
                        width: 28.0,
                        height: 28.0,
                    },
                    sensitivity: 0.78,
                },
                RegionOfInterest {
                    name: "porch".to_string(),
                    bounds: Rect {
                        x: 70.0,
                        y: 9.0,
                        width: 20.0,
                        height: 18.0,
                    },
                    sensitivity: 0.82,
                },
            ],
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct EdgeVisionPipeline {
    frame_index: u64,
    time: f32,
    previous_frame: Vec<u8>,
    tracks: Vec<Track>,
    next_track_id: u32,
    calibration: CameraCalibration,
    targets: Vec<SyntheticTarget>,
}

impl Default for EdgeVisionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl EdgeVisionPipeline {
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        Self {
            frame_index: 0,
            time: 0.0,
            previous_frame: vec![18; FRAME_PIXELS],
            tracks: Vec::new(),
            next_track_id: 1,
            calibration: CameraCalibration::default(),
            targets: vec![
                SyntheticTarget {
                    class_name: "person",
                    origin: Point { x: 61.0, y: 30.0 },
                    radius: Point { x: 17.0, y: 9.0 },
                    size: Point { x: 7.0, y: 14.0 },
                    phase: 0.2,
                    speed: 1.0,
                    intensity: 218,
                },
                SyntheticTarget {
                    class_name: "vehicle",
                    origin: Point { x: 24.0, y: 38.0 },
                    radius: Point { x: 16.0, y: 5.0 },
                    size: Point { x: 18.0, y: 8.0 },
                    phase: 1.9,
                    speed: 0.62,
                    intensity: 176,
                },
                SyntheticTarget {
                    class_name: "package",
                    origin: Point { x: 80.0, y: 20.0 },
                    radius: Point { x: 5.0, y: 3.0 },
                    size: Point { x: 6.0, y: 5.0 },
                    phase: 2.7,
                    speed: 0.34,
                    intensity: 142,
                },
            ],
        }
    }

    pub fn tick(&mut self, dt_ms: f32) {
        self.time += dt_ms / 1000.0;
        self.frame_index += 1;
    }

    pub fn snapshot_json(&mut self) -> String {
        let snapshot = self.snapshot();
        serde_json::to_string(&snapshot).unwrap_or_else(|_| "{}".to_string())
    }
}

impl EdgeVisionPipeline {
    pub fn snapshot(&mut self) -> PipelineSnapshot {
        let frame = self.render_frame();
        let motion_heatmap = diff_frames(&frame, &self.previous_frame);
        let detections = self.detect(&motion_heatmap);
        self.update_tracks(&detections);
        self.previous_frame = frame.clone();

        let latency = self.estimate_latency(detections.len(), self.tracks.len());
        PipelineSnapshot {
            frame_index: self.frame_index,
            timestamp_ms: self.time * 1000.0,
            width: FRAME_WIDTH,
            height: FRAME_HEIGHT,
            grayscale: frame,
            motion_heatmap,
            detections,
            tracks: self.tracks.clone(),
            latency,
            calibration: self.calibration.clone(),
        }
    }

    fn render_frame(&self) -> Vec<u8> {
        let mut frame = vec![0; FRAME_PIXELS];
        for y in 0..FRAME_HEIGHT {
            for x in 0..FRAME_WIDTH {
                let gradient = 18.0 + (y as f32 / FRAME_HEIGHT as f32) * 35.0;
                let texture = ((x as f32 * 0.37 + self.time * 1.7).sin() * 5.0)
                    + ((y as f32 * 0.29).cos() * 4.0);
                frame[y * FRAME_WIDTH + x] = (gradient + texture).clamp(0.0, 255.0) as u8;
            }
        }

        for region in &self.calibration.regions {
            draw_rect(&mut frame, region.bounds, 18);
        }

        for target in &self.targets {
            let bbox = target.bbox_at(self.time);
            fill_rect(&mut frame, bbox, target.intensity);
            draw_rect(&mut frame, bbox, 245);
        }

        frame
    }

    fn detect(&self, heatmap: &[u8]) -> Vec<Detection> {
        self.targets
            .iter()
            .map(|target| {
                let bbox = target.bbox_at(self.time);
                let centroid = bbox.center();
                let motion_score = average_motion(heatmap, bbox);
                let region = self
                    .calibration
                    .regions
                    .iter()
                    .find(|region| region.bounds.contains(centroid))
                    .map(|region| region.name.clone());
                let sensitivity = region
                    .as_ref()
                    .and_then(|name| {
                        self.calibration
                            .regions
                            .iter()
                            .find(|candidate| candidate.name == *name)
                    })
                    .map_or(0.62, |region| region.sensitivity);

                Detection {
                    class_name: target.class_name.to_string(),
                    confidence: (0.52 + motion_score * 0.004 + sensitivity * 0.28).min(0.99),
                    bbox,
                    centroid,
                    motion_score,
                    region,
                }
            })
            .filter(|detection| detection.confidence >= 0.55)
            .collect()
    }

    fn update_tracks(&mut self, detections: &[Detection]) {
        let mut matched_track_ids = Vec::new();

        for detection in detections {
            let best = self
                .tracks
                .iter()
                .enumerate()
                .filter(|(_, track)| track.class_name == detection.class_name)
                .filter(|(_, track)| !matched_track_ids.contains(&track.id))
                .map(|(index, track)| (index, distance(track.centroid, detection.centroid)))
                .filter(|(_, distance)| *distance < 18.0)
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            if let Some((index, _)) = best {
                let previous = self.tracks[index].centroid;
                let track = &mut self.tracks[index];
                track.velocity = Point {
                    x: detection.centroid.x - previous.x,
                    y: detection.centroid.y - previous.y,
                };
                track.bbox = detection.bbox;
                track.centroid = detection.centroid;
                track.age_frames += 1;
                track.missed_frames = 0;
                track.confidence = detection.confidence;
                matched_track_ids.push(track.id);
            } else {
                let track = Track {
                    id: self.next_track_id,
                    class_name: detection.class_name.clone(),
                    bbox: detection.bbox,
                    centroid: detection.centroid,
                    velocity: Point { x: 0.0, y: 0.0 },
                    age_frames: 1,
                    missed_frames: 0,
                    confidence: detection.confidence,
                };
                self.next_track_id += 1;
                matched_track_ids.push(track.id);
                self.tracks.push(track);
            }
        }

        for track in &mut self.tracks {
            if !matched_track_ids.contains(&track.id) {
                track.missed_frames += 1;
            }
        }

        self.tracks.retain(|track| track.missed_frames <= 8);
    }

    fn estimate_latency(&self, detections: usize, tracks: usize) -> LatencyStats {
        let capture_ms = 2.3 + (self.time * 0.7).sin().abs() * 0.8;
        let preprocess_ms = 1.4 + FRAME_PIXELS as f32 / 8200.0;
        let detection_ms = 4.8 + detections as f32 * 1.35 + (self.time * 1.3).cos().abs();
        let tracking_ms = 0.9 + tracks as f32 * 0.42;
        let total_ms = capture_ms + preprocess_ms + detection_ms + tracking_ms;
        LatencyStats {
            capture_ms,
            preprocess_ms,
            detection_ms,
            tracking_ms,
            total_ms,
            fps: 1000.0 / total_ms,
        }
    }
}

fn diff_frames(current: &[u8], previous: &[u8]) -> Vec<u8> {
    current
        .iter()
        .zip(previous)
        .map(|(a, b)| a.abs_diff(*b).saturating_mul(4))
        .collect()
}

fn average_motion(heatmap: &[u8], rect: Rect) -> f32 {
    let x0 = rect.x.max(0.0) as usize;
    let y0 = rect.y.max(0.0) as usize;
    let x1 = (rect.x + rect.width).min(FRAME_WIDTH as f32) as usize;
    let y1 = (rect.y + rect.height).min(FRAME_HEIGHT as f32) as usize;
    let mut sum = 0u32;
    let mut count = 0u32;

    for y in y0..y1 {
        for x in x0..x1 {
            sum += heatmap[y * FRAME_WIDTH + x] as u32;
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        sum as f32 / count as f32
    }
}

fn distance(a: Point, b: Point) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

fn fill_rect(frame: &mut [u8], rect: Rect, value: u8) {
    let x0 = rect.x.max(0.0) as usize;
    let y0 = rect.y.max(0.0) as usize;
    let x1 = (rect.x + rect.width).min(FRAME_WIDTH as f32) as usize;
    let y1 = (rect.y + rect.height).min(FRAME_HEIGHT as f32) as usize;

    for y in y0..y1 {
        for x in x0..x1 {
            frame[y * FRAME_WIDTH + x] = value;
        }
    }
}

fn draw_rect(frame: &mut [u8], rect: Rect, value: u8) {
    let x0 = rect.x.max(0.0) as usize;
    let y0 = rect.y.max(0.0) as usize;
    let x1 = (rect.x + rect.width).min((FRAME_WIDTH - 1) as f32) as usize;
    let y1 = (rect.y + rect.height).min((FRAME_HEIGHT - 1) as f32) as usize;

    for x in x0..=x1 {
        frame[y0 * FRAME_WIDTH + x] = value;
        frame[y1 * FRAME_WIDTH + x] = value;
    }

    for y in y0..=y1 {
        frame[y * FRAME_WIDTH + x0] = value;
        frame[y * FRAME_WIDTH + x1] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_motion_and_tracks() {
        let mut pipeline = EdgeVisionPipeline::new();
        pipeline.tick(33.0);
        let first = pipeline.snapshot();
        pipeline.tick(33.0);
        let second = pipeline.snapshot();

        assert_eq!(first.width, FRAME_WIDTH);
        assert_eq!(first.grayscale.len(), FRAME_PIXELS);
        assert!(!second.detections.is_empty());
        assert!(!second.tracks.is_empty());
        assert!(second.latency.fps > 30.0);
    }

    #[test]
    fn associates_detection_with_roi() {
        let mut pipeline = EdgeVisionPipeline::new();
        for _ in 0..20 {
            pipeline.tick(33.0);
        }
        let snapshot = pipeline.snapshot();

        assert!(
            snapshot
                .detections
                .iter()
                .any(|detection| detection.region.is_some())
        );
    }

    #[test]
    fn stale_tracks_are_removed() {
        let mut pipeline = EdgeVisionPipeline::new();
        let detection = Detection {
            class_name: "person".to_string(),
            confidence: 0.9,
            bbox: Rect {
                x: 10.0,
                y: 10.0,
                width: 5.0,
                height: 10.0,
            },
            centroid: Point { x: 12.5, y: 15.0 },
            motion_score: 20.0,
            region: None,
        };

        pipeline.update_tracks(&[detection]);
        assert_eq!(pipeline.tracks.len(), 1);

        for _ in 0..9 {
            pipeline.update_tracks(&[]);
        }

        assert!(pipeline.tracks.is_empty());
    }
}
