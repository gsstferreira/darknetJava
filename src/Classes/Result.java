package Classes;

public class Result {

    public final String Label;
    public final float Confidence;
    public final ResultBox BoundingBox;

    public Result(String name, float conf, int X, int Y, int w, int h) {

        this.Label = name;
        this.Confidence = conf;
        this.BoundingBox = new ResultBox(X,Y,w,h);
    }
}
