package Classes;

public class Result {

    public String Label;
    public float Confidence;
    public ResultBox BoundingBox;

    public Result(String name, float conf, int X, int Y, int w, int h) {

        this.Label = name;
        this.Confidence = conf;
        this.BoundingBox = new ResultBox(X,Y,w,h);
    }
}
