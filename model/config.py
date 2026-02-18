

class YOLOConfig:

    def __init__(self):
        self.model = 'yolo12s.pt'
        self._confidence_threshold = 0.4
        self._iou_threshold = 0.45
        self._person_class_id = 0
        self._line_start = (0, 680)
        self._line_end = (1900, 680)

