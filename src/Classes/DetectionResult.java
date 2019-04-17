package Classes;

import java.util.ArrayList;
import java.util.List;

public class DetectionResult {

    public float ProcessTimeSecs;
    public int TotalDetections;
    public int ImageWidth;
    public int ImageHeight;
    public List<Result> Detections;

    public DetectionResult(float procTime, List<Result> detections,int w, int h) {

        this.ProcessTimeSecs = procTime;
        this.ImageHeight = h;
        this.ImageWidth = w;

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
