from model.yolo_model import YOLOModel
from model.config import YOLOConfig
import cv2
import numpy as np
from collections import OrderedDict

class PersonCounter:
    def __init__(self, max_disappeared=50):
        self.model = YOLOModel()
        self.config = YOLOConfig()
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.track_history = OrderedDict()
        self.object_side = OrderedDict()

    @staticmethod
    def _compute_centroid(rect):
        x1, y1, x2, y2 = rect
        c_x = int((x1 + x2) / 2.0)
        c_y = int((y1 + y2) / 2.0)
        return (c_x, c_y)

    @staticmethod
    def _point_side(point, line_start, line_end):
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        cross_product = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross_product >= 0:
            return "inside"
        return "outside"

    @staticmethod
    def _distance_matrix(a, b):
        a_np = np.array(a, dtype=float)
        b_np = np.array(b, dtype=float)
        return np.linalg.norm(a_np[:, np.newaxis] - b_np[np.newaxis, :], axis=2)

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.track_history[self.next_object_id] = [centroid]
        self.object_side[self.next_object_id] = self._point_side(
            centroid,
            self.config._line_start,
            self.config._line_end,
        )
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.track_history[object_id]
        del self.object_side[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            return self.objects
        
        input_centroids = []
        for rect in rects:
            input_centroids.append(self._compute_centroid(rect))

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        distance = self._distance_matrix(object_centroids, input_centroids)

        rows = distance.min(axis=1).argsort()
        cols = distance.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            object_id = object_ids[row]
            centroid = input_centroids[col]
            self.objects[object_id] = centroid
            self.disappeared[object_id] = 0
            self.track_history[object_id].append(centroid)
            self.object_side[object_id] = self._point_side(
                centroid,
                self.config._line_start,
                self.config._line_end,
            )

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(distance.shape[0])) - used_rows
        unused_cols = set(range(distance.shape[1])) - used_cols

        if distance.shape[0] >= distance.shape[1]:
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        else:
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects

    def get_counts(self):
        inside = 0
        outside = 0
        for side in self.object_side.values():
            if side == "inside":
                inside += 1
            else:
                outside += 1
        return inside, outside

    def process(self, frame: np.ndarray):
        predictions = self.model.predict(frame)
        rects = [pred.rect for pred in predictions]
        objects = self.update(rects)

        for pred in predictions:
            x1, y1, x2, y2 = pred.rect
            confidence = pred.confidence
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 200, 50), 2)
            cv2.putText(
                frame,
                f"person {confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (50, 200, 50),
                1,
            )

        for object_id, centroid in objects.items():
            side = self.object_side.get(object_id, "outside")
            color = (0, 255, 0) if side == "inside" else (0, 0, 255)
            cv2.circle(frame, centroid, 4, color, -1)
            cv2.putText(
                frame,
                f"ID {object_id} - {side}",
                (centroid[0] + 8, centroid[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        cv2.line(frame, self.config._line_start, self.config._line_end, (255, 255, 0), 2)

        inside_count, outside_count = self.get_counts()
        cv2.putText(
            frame,
            f"Dentro: {inside_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Fora: {outside_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        return frame



if __name__ == "__main__":
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("video2.mp4")
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    counter = PersonCounter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = counter.process(frame)
        cv2.imshow("Person Counter", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()    
    cv2.destroyAllWindows()
    print("fim..")
