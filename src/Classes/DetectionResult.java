package Classes;

import java.util.ArrayList;
import java.util.List;

public class DetectionResult {

    public float ProcessTimeSecs;
    public int TotalDetections;
    public List<Result> Detections;

    public DetectionResult(float procTime, List<Result> detections) {

        this.ProcessTimeSecs = procTime;

        if(detections == null) {
            this.Detections = new ArrayList<>();
            this.TotalDetections = 0;
        }
        else {
            this.Detections = detections;
            this.TotalDetections = detections.size();
        }
    }
}
