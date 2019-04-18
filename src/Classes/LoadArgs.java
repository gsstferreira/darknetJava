package Classes;

import Yolo.Enums.DataType;

public class LoadArgs {

    public int threads;
    public String[] paths;
    public String path;
    public int n;
    public int m;
    public String[] labels;
    public int h;
    public int w;
    public int outW;
    public int outH;
    public int nh;
    public int nw;
    public int numBoxes;
    public int min;
    public int max;
    public int size;
    public int classes;
    public int background;
    public int scale;
    public int center;
    public int coords;
    public float jitter;
    public float angle;
    public float aspect;
    public float saturation;
    public float exposure;
    public float hue;

    public Data d;
    public Image im;
    public Image resized;
    public DataType type;
    public Tree[] hierarchy;
}
