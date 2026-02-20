from dataclasses import dataclass
from typing import Dict, Mapping, Optional


@dataclass
class SideCounter:
    inside_frames: int = 0
    outside_frames: int = 0

    def observe(self, side: str) -> None:
        if side == "inside":
            self.inside_frames += 1
        else:
            self.outside_frames += 1

    @property
    def total_frames(self) -> int:
        return self.inside_frames + self.outside_frames


class SideAverages:
    def __init__(self) -> None:
        self._per_id: Dict[int, SideCounter] = {}

    # observa os frames dentro e fora de cada id
    #se já existir o id em questão atualiza a contagem de frames dentro e fora para cada id
    #se não cria um SideCounter para cada id novo observado
    def observe(self, object_side: Mapping[int, str]) -> None:
        for object_id, side in object_side.items():
            if object_id not in self._per_id:
                self._per_id[object_id] = SideCounter()
            self._per_id[object_id].observe(side)

    # calcula a media de frames dentro e fora de cada id
    # retorna um dict com chave = id e values = valores de insid, outside e suas medias
    def per_id(self, fps: Optional[float] = None) -> Dict[int, dict]:
        snapshot: Dict[int, dict] = {}
        for object_id, stats in self._per_id.items():
            total = stats.total_frames
            if total == 0:
                inside_ratio = 0.0
                outside_ratio = 0.0
            else:
                inside_ratio = stats.inside_frames / total
                outside_ratio = stats.outside_frames / total

            item = {
                "inside_frames": stats.inside_frames,
                "outside_frames": stats.outside_frames,
                "inside_ratio": inside_ratio,
                "outside_ratio": outside_ratio,
            }

            if fps and fps > 0:
                item["inside_seconds"] = stats.inside_frames / fps
                item["outside_seconds"] = stats.outside_frames / fps

            snapshot[object_id] = item

        return snapshot

    def overall(self, fps: Optional[float] = None) -> dict:
        if not self._per_id:
            result = {
                "ids_count": 0,
                "avg_inside_frames_per_id": 0.0,
                "avg_outside_frames_per_id": 0.0,
                "avg_inside_ratio_per_id": 0.0,
                "avg_outside_ratio_per_id": 0.0,
            }
            if fps and fps > 0:
                result["avg_inside_seconds_per_id"] = 0.0
                result["avg_outside_seconds_per_id"] = 0.0
            return result

        ids_count = len(self._per_id)
        inside_frames_sum = sum(stats.inside_frames for stats in self._per_id.values())
        outside_frames_sum = sum(stats.outside_frames for stats in self._per_id.values())

        ratio_sum_inside = 0.0
        ratio_sum_outside = 0.0
        for stats in self._per_id.values():
            total = stats.total_frames
            if total == 0:
                continue
            ratio_sum_inside += stats.inside_frames / total
            ratio_sum_outside += stats.outside_frames / total

        result = {
            "ids_count": ids_count,
            "avg_inside_frames_per_id": inside_frames_sum / ids_count,
            "avg_outside_frames_per_id": outside_frames_sum / ids_count,
            "avg_inside_ratio_per_id": ratio_sum_inside / ids_count,
            "avg_outside_ratio_per_id": ratio_sum_outside / ids_count,
        }

        if fps and fps > 0:
            result["avg_inside_seconds_per_id"] = result["avg_inside_frames_per_id"] / fps
            result["avg_outside_seconds_per_id"] = result["avg_outside_frames_per_id"] / fps

        return result