package Yolo.Layers;

import Classes.Arrays.DetectionArray;
import Classes.Arrays.FloatArray;
import Classes.Box;
import Classes.Arrays.IntArray;
import Classes.Layer;
import Classes.Network;
import Classes.Tree;
import Tools.Buffers;
import Tools.Util;
import Yolo.Enums.LayerType;


public class RegionLayer extends Layer {

    public RegionLayer(int batch, int w, int h, int n, int classes, int coords) {

        this.type = LayerType.REGION;

        this.n = n;
        this.batch = batch;
        this.h = h;
        this.w = w;
        this.c = n*(classes + coords + 1);
        this.outW = this.w;
        this.outH = this.h;
        this.outC = this.c;
        this.classes = classes;
        this.coords = coords;
        this.cost = new FloatArray(1);
        this.biases = new FloatArray(n*2);
        this.biasUpdates = new FloatArray(n*2);
        this.outputs = h*w*n*(classes + coords + 1);
        this.inputs = this.outputs;
        this.truths = 30*(this.coords + 1);
        this.delta = new FloatArray(batch*this.outputs);
        this.output = new FloatArray(batch*this.outputs);
        int i;

        for(i = 0; i < n*2; ++i){

            this.biases.put(i,0.5f);
        }
        System.out.println("detection");
    }

    public void resize(int w, int h) {

        this.w = w;
        this.h = h;

        this.outputs = h*w*this.n*(this.classes + this.coords + 1);
        this.inputs = this.outputs;

        this.output = Buffers.realloc(this.output, this.batch*this.outputs);
        this.delta = Buffers.realloc(this.delta, this.batch*this.outputs);

    }

    public Box getRegionBox(FloatArray x, FloatArray biases, int n, int index, int i, int j, int w, int h, int stride) {

        Box b = new Box();

        b.x = (i + x.get(index)) / w;
        b.y = (j + x.get(index + stride)) / h;
        b.w = (float) Math.exp(x.get(index + 2*stride)) * biases.get(2*n) / w;
        b.h = (float) Math.exp(x.get(index + 3*stride)) * biases.get(2*n+1) / h;
        return b;
    }

    public float deltaRegionBox(Box truth, FloatArray x, FloatArray biases, int n, int index, int i, int j, int w, int h, FloatArray delta, float scale, int stride) {

        Box pred = getRegionBox(x, biases, n, index, i, j, w, h, stride);
        float iou = Box.boxIou(pred, truth);

        float tx = (truth.x*w - i);
        float ty = (truth.y*h - j);
        float tw = (float) Math.log(truth.w*w / biases.get(2*n));
        float th = (float) Math.log(truth.h*h / biases.get(2*n + 1));

        delta.put(index,scale * (tx - x.get(index)));
        delta.put(index + stride,scale * (ty - x.get(index + stride)));
        delta.put(index + 2*stride,scale * (tw - x.get(index + 2*stride)));
        delta.put(index + 3*stride,scale * (th - x.get(index + 3*stride)));

        return iou;
    }

    public void deltaRegionMask(FloatArray truth, FloatArray x, int n, int index, FloatArray delta, int stride, int scale) {

        int i;
        for(i = 0; i < n; ++i){

            delta.put(index + i*stride,scale*(truth.get(i) - x.get(index + i*stride)));
        }
    }

    public void deltaRegionClass(FloatArray output, FloatArray delta, int index, int clas, int classes, Tree hier, float scale, int stride, FloatArray avg_cat, int tag) {

        int i, n;
        if(hier != null){
            float pred = 1;
            while(clas >= 0){

                pred *= output.get(index + stride*clas);
                int g = hier.group[clas];
                int offset = hier.groupOffset[g];
                for(i = 0; i < hier.groupOffset[g]; ++i){

                    delta.put(index + stride*(offset + i),scale * (0 - output.get(index + stride*(offset + i))));
                }

                delta.put(index + stride*clas,scale * (1 - output.get(index + stride*clas)));
                clas = hier.parent[clas];
            }

            avg_cat.put(0,avg_cat.get(0) + pred);
        }
        else {
            if (delta.get(index) != 0 && tag != 0){

                delta.put(index + stride*clas,scale * (1 - output.get(index + stride*clas)));
                return;
            }
            for(n = 0; n < classes; ++n){

                delta.put(index + stride*n,scale * (((n == clas)?1 : 0) - output.get(index + stride*n)));

                if(n == clas) {
                    avg_cat.put(0,avg_cat.get(0) + output.get(index + stride*n));
                }
            }
        }
    }

    public float logit(float x) {

        return (float) Math.log(x/(1.0 - x));
    }

    public float tisnan(float x) {

        return (x != x) ? 1 : 0;
    }

    public int entryIndex(int batch, int location, int entry) {

        int n =   location / (this.w*this.h);
        int loc = location % (this.w*this.h);
        return batch*this.outputs + n*this.w*this.h*(this.coords+this.classes+1) + entry*this.w*this.h + loc;
    }

    public void forward(Network net) {

        int i,j,b,t,n;

        Buffers.copy(net.input,net.output,this.outputs*this.batch);
        this.delta.setValue(0,outputs*batch);

        if(net.train == 0) {
            return;
        }

        float avg_iou = 0;
        float recall = 0;
        float avg_cat = 0;
        float avg_obj = 0;
        float avg_anyobj = 0;
        int count = 0;
        int class_count = 0;
        this.cost = null;

        for (b = 0; b < this.batch; ++b) {
            if(this.softmaxTree != null){
                int onlyclass = 0;
                for(t = 0; t < 30; ++t){

                    FloatArray truthBox = net.truth.offsetNew(t*(coords + 1) + b*this.truths);
                    Box truth = Box.floatToBox(truthBox, 1);

                    if(truth.x == 0) {
                        break;
                    }

                    int clas = (int) net.truth.get(t*(coords + 1) + b*truths + coords);
                    float maxp = 0;
                    int maxi = 0;
                    if(truth.x > 100000 && truth.y > 100000){
                        for(n = 0; n < this.n*this.w*this.h; ++n){
                            int class_index = entryIndex(b, n, this.coords + 1);
                            int obj_index = entryIndex(b, n, this.coords);
                            float scale =  output.get(obj_index);

                            delta.put(obj_index, noObjectScale * ( - this.output.get(obj_index)));

                            FloatArray fb = this.output.offsetNew(class_index);
                            float p = scale * softmaxTree.getHierarchyprobability(fb, clas, this.w*this.h);

                            if(p > maxp){
                                maxp = p;
                                maxi = n;
                            }
                        }

                        int class_index = entryIndex(b, maxi, coords + 1);
                        int obj_index = entryIndex(b, maxi, coords);

                        FloatArray fbb = new FloatArray(1);
                        fbb.put(0,avg_cat);

                        deltaRegionClass(output,delta, class_index, clas,classes,softmaxTree,classScale, w*h, fbb, (softmax == 0) ? 1 : 0);
                        avg_cat = fbb.get(0);

                        if(output.get(obj_index) < .3f) {

                            delta.put(obj_index,objectScale * (.3f - output.get(obj_index)));
                        }
                        else  {

                            delta.put(obj_index,0);
                        }

                        delta.put(obj_index,0);
                        ++class_count;
                        onlyclass = 1;
                        break;
                    }
                }
                if(onlyclass != 0) {
                    continue;
                }
            }
            for (j = 0; j < this.h; ++j) {
                for (i = 0; i < this.w; ++i) {
                    for (n = 0; n < this.n; ++n) {

                        int Box_index = entryIndex(b, n*this.w*this.h + j*this.w + i, 0);
                        Box pred = getRegionBox(this.output, this.biases, n, Box_index, i, j, this.w, this.h, this.w*this.h);
                        float best_iou = 0;

                        for(t = 0; t < 30; ++t){

                            FloatArray truthBox = net.truth.offsetNew(t*(coords + 1) + b*this.truths);
                            Box truth = Box.floatToBox(truthBox, 1);

                            if(truth.x != 0) {
                                break;
                            }
                            float iou = Box.boxIou(pred, truth);
                            if (iou > best_iou) {
                                best_iou = iou;
                            }
                        }
                        int obj_index = entryIndex(b, n*this.w*this.h + j*this.w + i, this.coords);
                        avg_anyobj += this.output.get(obj_index);

                        delta.put(obj_index,this.noObjectScale * (0 - this.output.get(obj_index)));

                        if(this.background != 0) {

                            delta.put(obj_index,this.noObjectScale * (1 - this.output.get(obj_index)));
                        }

                        if (best_iou > this.thresh) {
                            delta.put(obj_index,0);
                        }

                        if(net.seen.get(0) < 12800){

                            Box truth = new Box();
                            truth.x = (i + .5f)/this.w;
                            truth.y = (j + .5f)/this.h;
                            truth.w = this.biases.get(2*n)/this.w;
                            truth.h = this.biases.get(2*n+1)/this.h;

                            deltaRegionBox(truth, output, biases, n, Box_index, i, j, w, h, delta, .01f, w*h);
                        }
                    }
                }
            }
            for(t = 0; t < 30; ++t){

                FloatArray truthBox = net.truth.offsetNew(t*(coords + 1) + b*this.truths);
                Box truth = Box.floatToBox(truthBox, 1);

                if(truth.x == 0) {
                    break;
                }
                float best_iou = 0;
                int best_n = 0;
                i = (int) (truth.x * this.w);
                j = (int) (truth.y * this.h);

                Box truth_shift = new Box(truth);
                truth_shift.x = 0;
                truth_shift.y = 0;
                for(n = 0; n < this.n; ++n){
                    int Box_index = entryIndex(b, n*this.w*this.h + j*this.w + i, 0);
                    Box pred = getRegionBox(this.output, this.biases, n, Box_index, i, j, this.w, this.h, this.w*this.h);

                    if(this.biasMatch != 0){
                        pred.w = this.biases.get(2*n)/this.w;
                        pred.h = this.biases.get(2*n+1)/this.h;
                    }
                    pred.x = 0;
                    pred.y = 0;
                    float iou = Box.boxIou(pred, truth_shift);
                    if (iou > best_iou){
                        best_iou = iou;
                        best_n = n;
                    }
                }

                int Box_index = entryIndex(b, best_n*w*h + j*w + i, 0);
                float iou = deltaRegionBox(truth, output, biases, best_n, Box_index, i, j, w, h, delta, coordScale *  (2 - truth.w*truth.h), this.w*this.h);

                if(this.coords > 4){
                    int mask_index = entryIndex(b, best_n*this.w*this.h + j*this.w + i, 4);

                    FloatArray truthMask = net.truth.offsetNew(t*(coords + 1) + b*this.truths + 5);
                    deltaRegionMask(truthMask, this.output, this.coords - 4, mask_index, this.delta, this.w*this.h, (int) this.maskScale);
                }
                if(iou > .5f) recall += 1;
                avg_iou += iou;
                int obj_index = entryIndex(b, best_n*this.w*this.h + j*this.w + i, this.coords);

                avg_obj += this.output.get(obj_index);

                delta.put(obj_index,objectScale * (1 - output.get(obj_index)));

                if (this.rescore != 0) {

                    delta.put(obj_index,objectScale * (iou - output.get(obj_index)));
                }

                if(this.background != 0){

                    delta.put(obj_index,objectScale * (0 - output.get(obj_index)));
                }

                int clas = (int) net.truth.get(t*(this.coords + 1) + b*this.truths + this.coords);

                if (this.map != null) {
                    clas = this.map.get(clas);
                }

                int class_index = entryIndex(b, best_n*this.w*this.h + j*this.w + i, this.coords + 1);

                FloatArray fbb = new FloatArray(1);
                fbb.put(0,avg_cat);

                deltaRegionClass(output, delta, class_index, clas, classes, softmaxTree, classScale, w*h, fbb, (softmax == 0) ? 1 : 0);
                avg_cat = fbb.get(0);
                ++count;
                ++class_count;
            }
        }

        cost.put(0, (float)Math.pow(Util.magArray(delta, outputs * batch), 2));

        String s = String.format("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n",
                avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(w*h*this.n*batch), recall/count, count);
        System.out.print(s);
    }

    @SuppressWarnings("EmptyMethod")
    public void backward(Network net) {

    }

    public void correctRegionBoxes(DetectionArray dets, int n, int w, int h, int netw, int neth, int relative) {

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

    public void getRegionDetections(int w, int h, int netw, int neth, float thresh, IntArray map, float tree_thresh, int relative, DetectionArray dets) {

        int i,j,n,z;
        FloatArray predictions = this.output;
        if (this.batch == 2) {
            FloatArray flip = this.output.offsetNew(this.outputs);

            for (j = 0; j < this.h; ++j) {
                for (i = 0; i < this.w/2; ++i) {
                    for (n = 0; n < this.n; ++n) {
                        for(z = 0; z < this.classes + this.coords + 1; ++z){
                            int i1 = z*this.w*this.h*this.n + n*this.w*this.h + j*this.w + i;
                            int i2 = z*this.w*this.h*this.n + n*this.w*this.h + j*this.w + (this.w - i - 1);

                            float swap = flip.get(i1);
                            flip.put(i1,flip.get(i2));
                            flip.put(i2,swap);

                            if(z == 0){

                                flip.put(i1, - flip.get(i1));
                                flip.put(i2, - flip.get(i2));
                            }
                        }
                    }
                }
            }
            for(i = 0; i < this.outputs; ++i){


                this.output.put(i,(this.output.get(i) + flip.get(i))/2.0f);
            }
        }
        for (i = 0; i < this.w*this.h; ++i){
            int row = i / this.w;
            int col = i % this.w;
            for(n = 0; n < this.n; ++n){
                int index = n*this.w*this.h + i;
                for(j = 0; j < this.classes; ++j){
                    dets.get(index).prob[j] = 0;
                }

                int obj_index  = entryIndex(0, n*this.w*this.h + i, this.coords);
                int Box_index  = entryIndex(0, n*this.w*this.h + i, 0);
                int mask_index = entryIndex(0, n*this.w*this.h + i, 4);

                float scale = (this.background != 0) ? 1 : predictions.get(obj_index);
                dets.get(index).bBox = getRegionBox(predictions, this.biases, n, Box_index, col, row, this.w, this.h, this.w*this.h);
                dets.get(index).objectness = scale > thresh ? scale : 0;

                if(dets.get(index).mask != null){
                    for(j = 0; j < this.coords - 4; ++j){
                        dets.get(index).mask[j] = this.output.get(mask_index + j*this.w*this.h);
                    }
                }

                int class_index = entryIndex(0, n*this.w*this.h + i, this.coords + ((this.background == 0) ? 1 : 0));
                if(this.softmaxTree != null){

                    FloatArray fb = predictions.offsetNew(class_index);
                    softmaxTree.hierarchyPredictions(fb, this.classes, false, this.w*this.h);

                    if(map != null){
                        for(j = 0; j < 200; ++j){
                            int class_index2 = entryIndex(0, n*this.w*this.h + i, this.coords + 1 + map.get(j));
                            float prob = scale*predictions.get(class_index2);
                            dets.get(index).prob[j] = (prob > thresh) ? prob : 0;
                        }
                    }
                    else {
                        int j1 =  softmaxTree.hierarchyTopPredictions(fb,tree_thresh, this.w*this.h);
                        dets.get(index).prob[j1] = (scale > thresh) ? scale : 0;
                    }
                }
                else {
                    if(dets.get(index).objectness != 0){
                        for(j = 0; j < this.classes; ++j){

                            int class_index3 = entryIndex(0, n*this.w*this.h + i, this.coords + 1 + j);
                            float prob = scale*predictions.get(class_index3);
                            dets.get(index).prob[j] = (prob > thresh) ? prob : 0;
                        }
                    }
                }
            }
        }
        correctRegionBoxes(dets, this.w*this.h*this.n, w, h, netw, neth, relative);
    }
}
