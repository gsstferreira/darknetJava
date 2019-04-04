package Classes;

import java.nio.FloatBuffer;
import java.util.*;


public class Box {
    
    private static class DBox {
        
        float dx;
        float dy;
        float dw;
        float dh;

        public DBox(){}
        public DBox(float X, float Y, float W, float H) {

            this.dx = X;
            this.dy = Y;
            this.dw = W;
            this.dh = H;
        }
    }
    
    public float x;
    public float y;
    public float w;
    public float h;

    private static final double log2 = Math.log(2);

    public Box(){}
    public Box(float X, float Y, float W, float H) {

        this.x = X;
        this.y = Y;
        this.h = H;
        this.w = W;
    }

    public static Box floatToBox(FloatBuffer f, int stride) {

        Box b = new Box();
        b.x = f.get(0);
        b.y = f.get(stride);
        b.w = f.get(2*stride);
        b.h = f.get(3*stride);
        
        return b;
    }

    public static Box floatToBox(float[] f, int stride) {

        Box b = new Box();
        b.x = f[0];
        b.y = f[stride];
        b.w = f[2*stride];
        b.h = f[3*stride];

        return b;
    }

    private static void sortNms(Detection[] detections, int size) {

        Arrays.parallelSort(detections,0,size,(pa, pb) -> {

            float diff;
            if (pb.sortClass >= 0) {
                diff = pa.prob[pb.sortClass] - pb.prob[pb.sortClass];
            } else {
                diff = pa.objectness - pb.objectness;
            }

            if (diff < 0) {
                return 1;
            } else if (diff > 0) {
                return -1;
            } else {
                return 0;
            }
        });
    }
    
    public static void doNmsObj(Detection[] dets, int total, int classes, float thresh) {

        int i, j, k;
        k = total-1;
        for(i = 0; i <= k; ++i){
            if(dets[i].objectness == 0){
                Detection swap = dets[i];
                dets[i] = dets[k];
                dets[k] = swap;
                --k;
                --i;
            }
        }
        total = k+1;

        for(i = 0; i < total; ++i){
            dets[i].sortClass = -1;
        }

        sortNms(dets,total);

        for(i = 0; i < total; ++i){
            if(dets[i].objectness == 0) {
                continue;
            }
            Box a = dets[i].bbox;
            for(j = i+1; j < total; ++j){
                if(dets[j].objectness == 0) {
                    continue;
                }
                Box b = dets[j].bbox;
                if (boxIou(a,b) > thresh){
                    dets[j].objectness = 0;
                    for(k = 0; k < classes; ++k){
                        dets[j].prob[k] = 0;
                    }
                }
            }
        }
    }

    public static void doNmsSort(Detection[] dets, int total, int classes, float thresh) {

        int i, j, k;
        k = total-1;
        for(i = 0; i <= k; ++i){
            if(dets[i].objectness == 0){
                Detection swap = dets[i];
                dets[i] = dets[k];
                dets[k] = swap;
                --k;
                --i;
            }
        }
        total = k+1;

        for(k = 0; k < classes; ++k){
            for(i = 0; i < total; ++i){
                dets[i].sortClass = k;
            }

            sortNms(dets,total);

            for(i = 0; i < total; ++i){
                if(dets[i].prob[k] == 0) {
                    continue;
                }
                Box a = dets[i].bbox;
                for(j = i+1; j < total; ++j){
                    Box b = dets[j].bbox;
                    if (boxIou(a,b) > thresh){
                        dets[j].prob[k] = 0;
                    }
                }
            }
        }
    }

    private static DBox derivative(Box a, Box b) {
        
        DBox d = new DBox();
        d.dx = 0;
        d.dw = 0;
        float l1 = a.x - a.w/2;
        float l2 = b.x - b.w/2;
        if (l1 > l2){
            d.dx -= 1;
            d.dw += .5;
        }
        float r1 = a.x + a.w/2;
        float r2 = b.x + b.w/2;
        if(r1 < r2){
            d.dx += 1;
            d.dw += .5;
        }
        if (l1 > r2) {
            d.dx = -1;
            d.dw = 0;
        }
        if (r1 < l2){
            d.dx = 1;
            d.dw = 0;
        }

        d.dy = 0;
        d.dh = 0;
        float t1 = a.y - a.h/2;
        float t2 = b.y - b.h/2;
        if (t1 > t2){
            d.dy -= 1;
            d.dh += .5;
        }
        float b1 = a.y + a.h/2;
        float b2 = b.y + b.h/2;
        if(b1 < b2){
            d.dy += 1;
            d.dh += .5;
        }
        if (t1 > b2) {
            d.dy = -1;
            d.dh = 0;
        }
        if (b1 < t2){
            d.dy = 1;
            d.dh = 0;
        }
        return d;
    }

    private static float overlap(float x1, float w1, float x2, float w2) {
        
        float l1 = x1 - w1/2;
        float l2 = x2 - w2/2;
        float left = l1 > l2 ? l1 : l2;
        float r1 = x1 + w1/2;
        float r2 = x2 + w2/2;
        float right = r1 < r2 ? r1 : r2;
        
        return right - left;
    }

    private static float boxIntersection(Box a, Box b) {
        
        float w = overlap(a.x, a.w, b.x, b.w);
        float h = overlap(a.y, a.h, b.y, b.h);
        if(w < 0 || h < 0) return 0;
        
        return w*h;
    }

    private static float boxUnion(Box a, Box b) {
        
        float i = boxIntersection(a, b);
        
        return a.w*a.h + b.w*b.h - i;
    }

    public static float boxIou(Box a, Box b) {
        
        return boxIntersection(a, b)/ boxUnion(a, b);
    }

    public static float boxRmse(Box a, Box b) {
        
        return (float) Math.sqrt(Math.pow(a.x-b.x, 2) + Math.pow(a.y-b.y, 2) + Math.pow(a.w-b.w, 2) + Math.pow(a.h-b.h, 2));
    }

    private static DBox dintersect(Box a, Box b) {
        
        float w = overlap(a.x, a.w, b.x, b.w);
        float h = overlap(a.y, a.h, b.y, b.h);
        DBox dover = derivative(a, b);
        DBox di = new DBox();

        di.dw = dover.dw*h;
        di.dx = dover.dx*h;
        di.dh = dover.dh*w;
        di.dy = dover.dy*w;

        return di;
    }

    private static DBox dunion(Box a, Box b) {

        DBox du = new DBox();

        DBox di = dintersect(a, b);
        du.dw = a.h - di.dw;
        du.dh = a.w - di.dh;
        du.dx = -di.dx;
        du.dy = -di.dy;

        return du;
    }

    private void testDunion() {

        Box a = new Box(0, 0, 1, 1);
        Box dxa = new Box(0+.0001f, 0, 1, 1);
        Box dya = new Box(0, 0+.0001f, 1, 1);
        Box dwa = new Box(0, 0, 1+.0001f, 1);
        Box dha = new Box(0, 0, 1, 1+.0001f);

        Box b = new Box(.5f, .5f, .2f, .2f);

        float inter =  boxUnion(a, b);
        float xinter = boxUnion(dxa, b);
        float yinter = boxUnion(dya, b);
        float winter = boxUnion(dwa, b);
        float hinter = boxUnion(dha, b);

    }

    private void testDintersect() {

        Box a = new Box(0, 0, 1, 1);
        Box dxa = new Box(0+.0001f, 0, 1, 1);
        Box dya = new Box(0, 0+.0001f, 1, 1);
        Box dwa = new Box(0, 0, 1+.0001f, 1);
        Box dha = new Box(0, 0, 1, 1+.0001f);

        Box b = new Box(.5f, .5f, .2f, .2f);

        DBox di = dintersect(a,b);

        float inter =  boxIntersection(a, b);
        float xinter = boxIntersection(dxa, b);
        float yinter = boxIntersection(dya, b);
        float winter = boxIntersection(dwa, b);
        float hinter = boxIntersection(dha, b);
        xinter = (xinter - inter)/(.0001f);
        yinter = (yinter - inter)/(.0001f);
        winter = (winter - inter)/(.0001f);
        hinter = (hinter - inter)/(.0001f);

    }

    public void testBox() {

        testDintersect();
        testDunion();

        Box a = new Box(0, 0, 1, 1);
        Box dxa = new Box(0+.00001f, 0, 1, 1);
        Box dya = new Box(0, 0+.00001f, 1, 1);
        Box dwa = new Box(0, 0, 1+.00001f, 1);
        Box dha = new Box(0, 0, 1, 1+.00001f);

        Box b = new Box(.5f, 0, .2f, .2f);


        float iou = boxIou(a,b);
        iou = (1-iou)*(1-iou);

        DBox d = diou(a, b);

        float xiou = boxIou(dxa, b);
        float yiou = boxIou(dya, b);
        float wiou = boxIou(dwa, b);
        float hiou = boxIou(dha, b);
        xiou = ((1-xiou)*(1-xiou) - iou)/(.00001f);
        yiou = ((1-yiou)*(1-yiou) - iou)/(.00001f);
        wiou = ((1-wiou)*(1-wiou) - iou)/(.00001f);
        hiou = ((1-hiou)*(1-hiou) - iou)/(.00001f);
    }

    public static DBox diou(Box a, Box b) {

        float u = boxUnion(a,b);
        float i = boxIntersection(a,b);
        DBox di = dintersect(a,b);
        DBox du = dunion(a,b);
        DBox dd = new DBox(0,0,0,0);

        if(i <= 0) {
            dd.dx = b.x - a.x;
            dd.dy = b.y - a.y;
            dd.dw = b.w - a.w;
            dd.dh = b.h - a.h;
            return dd;
        }

        dd.dx = 2*(float)Math.pow((1-(i/u)),1)*(di.dx*u - du.dx*i)/(u*u);
        dd.dy = 2*(float)Math.pow((1-(i/u)),1)*(di.dy*u - du.dy*i)/(u*u);
        dd.dw = 2*(float)Math.pow((1-(i/u)),1)*(di.dw*u - du.dw*i)/(u*u);
        dd.dh = 2*(float)Math.pow((1-(i/u)),1)*(di.dh*u - du.dh*i)/(u*u);
        return dd;
    }

    public  static void doNms(Box[] Boxes, float[][] probs, int total, int classes, float thresh) {

        int i, j, k;
        for(i = 0; i < total; ++i){
            int any = 0;
            for(k = 0; k < classes; ++k) {
                any = (any != 0 || (probs[i][k] > 0)) ? 1 : 0;
            }
            if(any == 0) {
                continue;
            }
            for(j = i+1; j < total; ++j){
                if (boxIou(Boxes[i], Boxes[j]) > thresh){
                    for(k = 0; k < classes; ++k){
                        if (probs[i][k] < probs[j][k]) probs[i][k] = 0;
                        else probs[j][k] = 0;
                    }
                }
            }
        }
    }

    public static Box encodeBox(Box b, Box anchor) {

        Box encode = new Box();
        encode.x = (b.x - anchor.x) / anchor.w;
        encode.y = (b.y - anchor.y) / anchor.h;
        encode.w = (float)(Math.log(b.w / anchor.w)/log2);
        encode.h = (float)(Math.log(b.h / anchor.h)/log2);
        return encode;
    }

    public static Box decodeBox(Box b, Box anchor) {

        Box decode = new Box();
        decode.x = b.x * anchor.w + anchor.x;
        decode.y = b.y * anchor.h + anchor.y;
        decode.w = (float)Math.pow(2., b.w) * anchor.w;
        decode.h = (float)Math.pow(2., b.h) * anchor.h;
        return decode;
    }
        
}
