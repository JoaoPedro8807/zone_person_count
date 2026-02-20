from dataclasses import dataclass
from model.config import YOLOConfig    
from ultralytics import YOLO       
import numpy as np 


@dataclass
class PredictData:
    rect: tuple[int, int, int, int]
    confidence: float
    track_id: int | None = None
        


class YOLOModel:
    def __init__(self):
        self.config = YOLOConfig()
        self.model = YOLO(self.config.model)


    def predict(self, frame: np.ndarray, **kwargs) -> list[PredictData]:
        confidence = kwargs.get("confidence", self.config._confidence_threshold)
        iou = kwargs.get("iou", self.config._iou_threshold)
        person_class_id = kwargs.get("class_id", self.config._person_class_id)

        results = self.model.predict(
            source=frame,
            conf=confidence,
            iou=iou,
            verbose=False,
        )

        detections: list[PredictData] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            class_id = int(box.cls[0].item())
            if class_id != person_class_id:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            detections.append(PredictData((int(x1), int(y1), int(x2), int(y2)), conf))

        return detections

    def track(self, frame: np.ndarray, **kwargs) -> list[PredictData]:
        confidence = kwargs.get("confidence", self.config._confidence_threshold)
        iou = kwargs.get("iou", self.config._iou_threshold)
        person_class_id = kwargs.get("class_id", self.config._person_class_id)
        tracker = kwargs.get("tracker", self.config._tracker)

        #TODO testar alguns parametros para melhorar o tracking
        #https://docs.ultralytics.com/pt/modes/track/#tracker-arguments

        results = self.model.track(
            source=frame,
            conf=confidence,
            iou=iou,
            classes=[person_class_id],
            tracker=tracker,
            persist=True,
            verbose=False,
        )

        tracks: list[PredictData] = []
        if not results:
            return tracks

        boxes = results[0].boxes
        if boxes is None:
            return tracks

        for box in boxes:
            if box.id is None:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            track_id = int(box.id[0].item())
            tracks.append(PredictData((int(x1), int(y1), int(x2), int(y2)), conf, track_id))

        return tracks
    

