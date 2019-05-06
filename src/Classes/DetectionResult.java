package Classes;

import java.util.ArrayList;
import java.util.List;

public class DetectionResult {

    public final float ProcessTimeSecs;
    public final float Threshold;
    public final int TotalDetections;
    public final int ImageWidth;
    public final int ImageHeight;
    public final List<Result> Detections;

    public DetectionResult(float procTime, float threshold, List<Result> detections,int w, int h) {

        this.ProcessTimeSecs = procTime;
        this.Threshold = threshold;
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
