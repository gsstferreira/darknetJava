package Yolo.Layers;

import Classes.Arrays.FloatArray;
import Classes.Arrays.IntArray;
import Classes.Box;
import Classes.Arrays.DetectionArray;
import Classes.Layer;
import Classes.Network;
import Tools.Blas;
import Tools.Buffers;
import Tools.Util;
import Yolo.Enums.Activation;
import Yolo.Enums.LayerType;

public class YoloLayer extends Layer {

    public YoloLayer(int batch, int w, int h, int n, int total, IntArray mask, int classes) {

        int i;
        this.type = LayerType.YOLO;

        this.n = n;
        this.total = total;
        this.batch = batch;
        this.h = h;
        this.w = w;
        this.c = n*(classes + 4 + 1);
        this.outW = this.w;
        this.outH = this.h;
        this.outC = this.c;
        this.classes = classes;
        this.cost = new FloatArray(1);
        this.biases = new FloatArray(total*2);

        if(mask != null) {
            this.mask = mask;
        }
        else{
            this.mask = new IntArray(n);
            for(i = 0; i < n; ++i){

                this.mask.set(i,i);
            }
        }
        this.biasUpdates = new FloatArray(n*2);
        this.outputs = h*w*n*(classes + 4 + 1);
        this.inputs = this.outputs;
        this.truths = 90*(4 + 1);
        this.delta = new FloatArray(batch*this.outputs);
        this.output = new FloatArray(batch*this.outputs);

        this.biases.setAll(0.5f,2*total);

        System.out.println("Yolo");
    }

    public void resize(int w, int h) {

        this.w = w;
        this.h = h;

        this.outputs = h*w*this.n*(this.classes + 4 + 1);
        this.inputs = this.outputs;

        this.output = Buffers.realloc(this.output, this.batch*this.outputs);
        this.delta = Buffers.realloc(this.delta, this.batch*this.outputs);
    }

    public static Box getYoloBox(FloatArray x, FloatArray biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride) {

        Box b = new Box();
        b.x = (i + x.get(index)) / lw;
        b.y = (j + x.get(index + stride)) / lh;
        b.w = (float) Math.exp(x.get(index + 2*stride)) * biases.get(2*n)   / w;
        b.h = (float) Math.exp(x.get(index + 3*stride)) * biases.get(2*n+1) / h;
        return b;
    }

    public static float deltaYoloBox(Box truth, FloatArray x, FloatArray biases, int n, int index, int i, int j,
                                     int lw, int lh, int w, int h, FloatArray delta, float scale, int stride) {

        Box pred = getYoloBox(x, biases, n, index, i, j, lw, lh, w, h, stride);
        float iou = Box.boxIou(pred, truth);

        float tx = (truth.x*lw - i);
        float ty = (truth.y*lh - j);
        float tw = (float) Math.log(truth.w*w / biases.get(2*n));
        float th = (float) Math.log(truth.h*h / biases.get(2*n + 1));

        delta.set(index,scale * (tx - x.get(index)));
        delta.set(index + stride,scale * (ty - x.get(index + stride)));
        delta.set(index + 2*stride,scale * (tw - x.get(index + 2*stride)));
        delta.set(index + 3*stride,scale * (th - x.get(index + 3*stride)));

        return iou;
    }


    public static void deltaYoloClass(FloatArray output, FloatArray delta, int index, int clas, int classes, int stride, FloatArray avg_cat) {

        int n;
        if (delta.get(index) != 0){

            delta.set(index + stride*clas,1 - output.get(index + stride*clas));

            if(avg_cat != null) {

                avg_cat.set(0,avg_cat.get(0) + output.get(index + stride*clas));
            }
            return;
        }

        for(n = 0; n < classes; ++n){

            delta.set(index + stride*n,((n == clas)?1 : 0) - output.get(index + stride*n));

            if(n == clas && avg_cat != null) {

                avg_cat.set(0,avg_cat.get(0) + output.get(index + stride*n));
            }
        }
    }

    public int entryIndex(int batch, int location, int entry) {

        final int wh = this.w*this.h;

        final int n =   location / wh;
        final int loc = location % wh;

        return batch*this.outputs + n*wh*(5 + this.classes) + entry*wh + loc;
    }

    public void forward(Network net) {

        net.input.copyInto(this.outputs*this.batch,this.output);
        this.delta.setAll(0,this.outputs*this.batch);

        final int wh = w * h;
        final int classesWh = (1 + this.classes) * wh;

        for (int b = 0; b < this.batch; ++b){
            for(int n = 0; n < this.n; ++n){

                final int index = entryIndex(b, n*wh, 0);

                FloatArray fb = this.output.offsetNew(index);
                Activation.activateArray(fb,2*wh, Activation.LOGISTIC);

                final int index2 = entryIndex(b, n*wh, 4);
                fb.offset(index2 - index);
                Activation.activateArray(fb, classesWh, Activation.LOGISTIC);
            }
        }

        if(net.train != 0) {
            float avg_iou = 0;
            float recall = 0;
            float recall75 = 0;
            float avg_cat = 0;
            float avg_obj = 0;
            float avg_anyobj = 0;
            int count = 0;
            int class_count = 0;

            this.cost.set(0,0);

            for (int b = 0; b < this.batch; ++b) {
                for (int j = 0; j < this.h; ++j) {
                    for (int i = 0; i < this.w; ++i) {
                        for (int n = 0; n < this.n; ++n) {
                            int Box_index = entryIndex(b, n*this.w*this.h + j*this.w + i, 0);
                            Box pred = getYoloBox(this.output, this.biases, this.mask.get(n), Box_index, i, j, this.w, this.h, net.w, net.h, this.w*this.h);
                            float best_iou = 0;
                            int best_t = 0;
                            for(int t = 0; t < this.maxBoxes; ++t){

                                FloatArray fb = net.truth.offsetNew(t*(4 + 1) + b*this.truths);

                                Box truth = Box.floatToBox(fb, 1);

                                if(truth.x == 0) {
                                    break;
                                }

                                float iou = Box.boxIou(pred, truth);
                                if (iou > best_iou) {
                                    best_iou = iou;
                                    best_t = t;
                                }
                            }
                            int obj_index = entryIndex(b, n*this.w*this.h + j*this.w + i, 4);
                            avg_anyobj += this.output.get(obj_index);

                            this.delta.set(obj_index,0 - this.output.get(obj_index));

                            if (best_iou > this.ignoreThresh) {

                                this.delta.set(obj_index,0);
                            }

                            if (best_iou > this.truthThresh) {

                                this.delta.set(obj_index,1 - this.output.get(obj_index));

                                int clas = (int) net.truth.get(best_t*(4 + 1) + b*this.truths + 4);

                                if (this.map != null) {
                                    clas = this.map.get(clas);
                                }

                                int class_index = entryIndex(b, n*this.w*this.h + j*this.w + i, 4 + 1);
                                deltaYoloClass(this.output, this.delta, class_index, clas, this.classes, this.w*this.h, null);

                                FloatArray fb = net.truth.offsetNew(best_t*(4 + 1) + b*this.truths);
                                Box truth = Box.floatToBox(fb, 1);

                                deltaYoloBox(truth, output, biases, mask.get(n), Box_index, i, j, w, h, net.w, net.h, delta, (2-truth.w*truth.h), w*h);
                            }
                        }
                    }
                }

                for(int t = 0; t < this.maxBoxes; ++t){

                    FloatArray fb = net.truth.offsetNew(t*(4 + 1) + b*this.truths);
                    Box truth = Box.floatToBox(fb, 1);

                    if(truth.x == 0) {
                        break;
                    }

                    float best_iou = 0;
                    int best_n = 0;
                    int i = (int) (truth.x * this.w);
                    int j = (int) (truth.y * this.h);

                    Box truth_shift = new Box(truth);
                    truth_shift.x = truth_shift.y = 0;
                    for(n = 0; n < this.total; ++n){
                        Box pred = new Box();
                        pred.w = this.biases.get(2*n)/net.w;
                        pred.h = this.biases.get(2*n+1)/net.h;
                        float iou = Box.boxIou(pred, truth_shift);
                        if (iou > best_iou){
                            best_iou = iou;
                            best_n = n;
                        }
                    }

                    int mask_n = Util.intIndex(this.mask, best_n, this.n);

                    if(mask_n >= 0){
                        int Box_index = entryIndex(b, mask_n*this.w*this.h + j*this.w + i, 0);
                        float iou = deltaYoloBox(truth, output, biases, best_n, Box_index, i, j, w, h, net.w, net.h, delta, (2-truth.w*truth.h), w*h);

                        int obj_index = entryIndex(b, mask_n*this.w*this.h + j*this.w + i, 4);
                        avg_obj += this.output.get(obj_index);

                        this.delta.set(obj_index,1 - this.output.get(obj_index));

                        int clas = (int) net.truth.get(t*(4 + 1) + b*this.truths + 4);

                        if (this.map != null) {
                            clas = this.map.get(clas);
                        }

                        int class_index = entryIndex(b, mask_n*this.w*this.h + j*this.w + i, 4 + 1);

                        FloatArray avg = new FloatArray(1);
                        avg.set(0,avg_cat);

                        deltaYoloClass(this.output, this.delta, class_index, clas, this.classes, this.w*this.h, avg);
                        avg_cat = avg.get(0);
                        ++count;
                        ++class_count;
                        if(iou > .5) recall += 1;
                        if(iou > .75) recall75 += 1;
                        avg_iou += iou;
                    }
                }
            }

            this.cost.set(0,(float) Math.pow(Util.magArray(this.delta, this.outputs * this.batch), 2));

            String s = String.format("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", net.index, avg_iou/count,
                    avg_cat/class_count, avg_obj/count, avg_anyobj/(this.w*this.h*this.n*this.batch), recall/count, recall75/count, count);

            System.out.print(s);
        }
    }

    public void backward(Network net) {

        Blas.axpyCpu(this.batch * this.inputs, 1, this.delta, 1, net.delta, 1);
    }

    public void correctYoloBoxes(DetectionArray dets, int n, int w, int h, int netw, int neth, int relative) {

        int i;
        int new_w;
        int new_h;

        if (((float)netw/w) < ((float)neth/h)) {
            new_w = netw;
            new_h = (h * netw)/w;
        }

        else {
            new_h = neth;
            new_w = (w * neth)/h;
        }

        for (i = 0; i < n; ++i){
            Box b = dets.get(i).bBox;
            b.x =  (b.x - (netw - new_w)/2.0f/netw) / ((float)new_w/netw);
            b.y =  (b.y - (neth - new_h)/2.0f/neth) / ((float)new_h/neth);
            b.w *= (float)netw/new_w;
            b.h *= (float)neth/new_h;

            if(relative == 0){
                b.x *= w;
                b.w *= w;
                b.y *= h;
                b.h *= h;
            }
            dets.get(i).bBox = b;
        }
    }

    public int numDetections(float thresh) {

        int i, n;
        int count = 0;
        for (i = 0; i < this.w*this.h; ++i){
            for(n = 0; n < this.n; ++n){
                int objIndex  = entryIndex(0, n*this.w*this.h + i, 4);

                float conf = this.output.get(objIndex);

                if(conf > thresh){
                    ++count;
                }
            }
        }
        return count;
    }

    private void avgFlippedYolo() {

        int i,j,n,z;
        FloatArray flip = this.output.offsetNew(this.outputs);
        for (j = 0; j < this.h; ++j) {
            for (i = 0; i < this.w/2; ++i) {
                for (n = 0; n < this.n; ++n) {
                    for(z = 0; z < this.classes + 4 + 1; ++z) {
                        int i1 = z*this.w*this.h*this.n + n*this.w*this.h + j*this.w + i;
                        int i2 = z*this.w*this.h*this.n + n*this.w*this.h + j*this.w + (this.w - i - 1);

                        float swap = flip.get(i1);
                        flip.set(i1, flip.get(i2));
                        flip.set(i2,swap);

                        if(z == 0){

                            flip.set(i1, - flip.get(i1));
                            flip.set(i2, - flip.get(i2));
                        }
                    }
                }
            }
        }
        for(i = 0; i < this.outputs; ++i) {

            this.output.set(i,(this.output.get(i) + flip.get(i))/2.0f);
        }
    }

    public int getYoloDetections(int w, int h, int netw, int neth, float thresh, IntArray map, int relative, DetectionArray dets) {

        FloatArray predictions = this.output;

        final int wh = this.w*this.h;

        if (this.batch == 2) {
            avgFlippedYolo();
        }

        int count = 0;
        for (int i = 0; i < wh; ++i){

            final int row = i / this.w;
            final int col = i % this.w;

            for(int n = 0; n < this.n; ++n){

                final int iNwh = n*wh + i;
                final int objIndex  = entryIndex(0, iNwh, 4);
                final int boxIndex  = entryIndex(0, iNwh, 0);

                final float objectness = predictions.get(objIndex);

                if(objectness > thresh) {

                    dets.get(count).bBox = getYoloBox(predictions, this.biases, this.mask.get(n), boxIndex, col, row, this.w, this.h, netw, neth, wh);
                    dets.get(count).objectness = objectness;
                    dets.get(count).classes = this.classes;

                    for(int j = 0; j < this.classes; ++j){

                        final int classIndex = entryIndex(0, iNwh, 5 + j);
                        final float prob = objectness*predictions.get(classIndex);
                        dets.get(count).prob[j] = (prob > thresh) ? prob : 0;
                    }
                    ++count;
                }
            }
        }
        correctYoloBoxes(dets, count, w, h, netw, neth, relative);
        return count;
    }
}
