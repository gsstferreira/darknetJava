package Layers;

import Classes.Box;
import Classes.Detection;
import Classes.Layer;
import Classes.Network;
import Enums.LayerType;
import Tools.Blas;
import Tools.BufferUtil;
import Tools.Rand;
import Tools.Util;
import org.lwjgl.BufferUtils;
import java.nio.FloatBuffer;

public class DetectionLayer extends Layer {

    public DetectionLayer(int batch, int inputs, int n, int side, int classes, int coords, int rescore) {

        this.type = LayerType.DETECTION;

        this.n = n;
        this.batch = batch;
        this.inputs = inputs;
        this.classes = classes;
        this.coords = coords;
        this.rescore = rescore;
        this.side = side;
        this.w = side;
        this.h = side;

        assert(side*side*((1 + this.coords)*this.n + this.classes) == inputs);

        this.cost = BufferUtils.createFloatBuffer(1);
        this.outputs = this.inputs;
        this.truths = this.side*this.side*(1+this.coords+this.classes);
        this.output = BufferUtils.createFloatBuffer(batch*this.outputs);
        this.delta = BufferUtils.createFloatBuffer(batch*this.outputs);
    }

    public void forward(Network net) {

        int locations = this.side*this.side;
        int i,j;

        BufferUtil.bufferCopy(net.input,output,outputs*batch);

        int b;
        if (this.softmax != 0){
            for(b = 0; b < this.batch; ++b){
                int index = b*this.inputs;
                for (i = 0; i < locations; ++i) {
                    int offset = i*this.classes;

                    FloatBuffer fb = BufferUtil.offsetBuffer(this.output,index + offset);
                    Blas.softmax(fb, this.classes, 1, 1, fb);
                }
            }
        }
        if(net.train != 0){

            float avg_iou = 0;
            float avg_cat = 0;
            float avg_allcat = 0;
            float avg_obj = 0;
            float avg_anyobj = 0;
            int count = 0;
            this.cost.put(0,0);

            int size = this.inputs * this.batch;

            BufferUtil.setBufferValue(this.delta,0,size);

            for (b = 0; b < this.batch; ++b){
                int index = b*this.inputs;
                for (i = 0; i < locations; ++i) {

                    int truth_index = (b*locations + i)*(1+this.coords+this.classes);

                    int is_obj = (int) net.truth.get(truth_index);

                    for (j = 0; j < this.n; ++j) {
                        int p_index = index + locations*this.classes + i*this.n + j;

                        float val = this.noobjectScale * ( -this.output.get(p_index));
                        this.delta.put(p_index,val);

                        val = this.cost.get(0) + (noobjectScale * (float)Math.pow(output.get(p_index),2));
                        this.cost.put(0,val);
                        avg_anyobj += this.output.get(p_index);
                    }

                    int best_index = -1;
                    float best_iou = 0;
                    float best_rmse = 20;

                    if (is_obj == 0){
                        continue;
                    }

                    int class_index = index + i*this.classes;
                    for(j = 0; j < this.classes; ++j) {

                        float val = this.classScale * (net.truth.get(truth_index+1+j) - this.output.get(class_index+j));
                        this.delta.put(class_index+j,val);

                        val = this.cost.get(0) + (float)(this.classScale * Math.pow(net.truth.get(truth_index+1+j) - this.output.get(class_index+j),2));
                        this.cost.put(0,val);

                        if(net.truth.get(truth_index + 1 + j) != 0) {
                            avg_cat += this.output.get(class_index+j);
                        }
                        avg_allcat += this.output.get(class_index+j);
                    }

                    FloatBuffer fb = BufferUtil.offsetBuffer(net.truth,truth_index + 1 + this.classes);
                    Box truth = Box.floatToBox(fb, 1);
                    truth.x /= this.side;
                    truth.y /= this.side;

                    for(j = 0; j < this.n; ++j){

                        int box_index = index + locations*(this.classes + this.n) + (i*this.n + j) * this.coords;

                        FloatBuffer fb2 = BufferUtil.offsetBuffer(this.output, box_index);
                        Box out = Box.floatToBox(fb2, 1);
                        out.x /= this.side;
                        out.y /= this.side;

                        if (this.sqrt != 0){
                            out.w = out.w*out.w;
                            out.h = out.h*out.h;
                        }

                        float iou  = Box.boxIou(out, truth);
                        float rmse = Box.boxRmse(out, truth);

                        if(best_iou > 0 || iou > 0){
                            if(iou > best_iou){
                                best_iou = iou;
                                best_index = j;
                            }
                        }
                        else{
                            if(rmse < best_rmse){
                                best_rmse = rmse;
                                best_index = j;
                            }
                        }
                    }

                    if(this.forced != 0){
                        if(truth.w*truth.h < .1){
                            best_index = 1;
                        }
                        else{
                            best_index = 0;
                        }
                    }
                    if(this.random != 0 && net.seen.get(0) < 64000){
                        best_index = Rand.randInt()%this.n;
                    }

                    int bIndex = index + locations*(this.classes + this.n) + (i*this.n + best_index) * this.coords;
                    int tbIndex = truth_index + 1 + this.classes;

                    fb = BufferUtil.offsetBuffer(this.output,bIndex);
                    Box out = Box.floatToBox(fb, 1);
                    out.x /= this.side;
                    out.y /= this.side;

                    if (this.sqrt != 0) {
                        out.w = out.w*out.w;
                        out.h = out.h*out.h;
                    }

                    float iou  = Box.boxIou(out, truth);
                    int p_index = index + locations*this.classes + i*this.n + best_index;

                    double val = this.cost.get(0) - (noobjectScale * Math.pow(output.get(p_index),2)) + (objectScale * Math.pow(1 - output.get(p_index),2));
                    this.cost.put(0,(float) val);
                    avg_obj += this.output.get(p_index);

                    this.delta.put(p_index,this.objectScale * (1.0f - this.output.get(p_index)));

                    if(this.rescore != 0){

                        this.delta.put(p_index,this.objectScale * (iou - this.output.get(p_index)));
                    }

                    delta.put(bIndex,coordScale*(net.truth.get(tbIndex) - output.get(bIndex)));
                    delta.put(bIndex + 1,coordScale*(net.truth.get(tbIndex + 1) - output.get(bIndex + 1)));

                    if(sqrt != 0) {
                        delta.put(bIndex + 2,coordScale*((float)Math.sqrt(net.truth.get(tbIndex + 2)) - output.get(bIndex + 2)));
                        delta.put(bIndex + 3,coordScale*((float)Math.sqrt(net.truth.get(tbIndex + 3)) - output.get(bIndex + 3)));
                    }
                    else {
                        delta.put(bIndex + 2,coordScale*(net.truth.get(tbIndex + 2) - output.get(bIndex + 2)));
                        delta.put(bIndex + 3,coordScale*(net.truth.get(tbIndex + 3) - output.get(bIndex + 3)));
                    }


                    this.cost.put(0,cost.get(0) + (float)Math.pow(1 - iou,2));
                    avg_iou += iou;
                    ++count;
                }
            }

            this.cost.put(0,(float)Math.pow(Util.magArray(this.delta, this.outputs * this.batch), 2));
        }
    }

    public void backward(Network net) {

        Blas.axpyCpu(this.batch*this.inputs, 1, this.delta, 1, net.delta, 1);
    }

    public void getDetectionDetections(int w, int h, float thresh, Detection[] dets) {

        int i,j,n;
        FloatBuffer predictions = this.output;

        for (i = 0; i < this.side*this.side; ++i){
            int row = i / this.side;
            int col = i % this.side;
            for(n = 0; n < this.n; ++n){
                int index = i*this.n + n;
                int p_index = this.side*this.side*this.classes + i*this.n + n;

                float scale = predictions.get(p_index);

                int box_index = this.side*this.side*(this.classes + this.n) + (i*this.n + n)*4;
                Box b = new Box();
                b.x = (predictions.get(box_index) + col) / this.side * w;
                b.y = (predictions.get(box_index + 1) + row) / this.side * h;
                b.w = (float)Math.pow(predictions.get(box_index + 2), ((this.sqrt != 0) ? 2:1)) * w;
                b.h = (float)Math.pow(predictions.get(box_index + 3), ((this.sqrt != 0) ? 2:1)) * h;

                dets[index].bbox = b;
                dets[index].objectness = scale;

                for(j = 0; j < this.classes; ++j){
                    int class_index = i*this.classes;
                    float prob = scale*predictions.get(class_index+j);
                    dets[index].prob[j] = (prob > thresh) ? prob : 0;
                }
            }
        }
    }
}
