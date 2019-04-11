package Classes;

import java.util.ArrayList;
import java.util.List;

public class DetectionResult {

    public float ProcessTimeSecs;
    public int TotalDetections;
    public List<Result> Detections;

    public DetectionResult(float procTime,int total, List<Result> detections) {

        this.ProcessTimeSecs = procTime;
        this.TotalDetections = total;

        if(detections == null) {
            this.Detections = new ArrayList<>();
        }
        else {
            this.Detections = detections;
        }
    }
}
