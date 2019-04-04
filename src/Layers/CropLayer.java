package Layers;

import Classes.Image;
import Classes.Layer;
import Classes.Network;
import Enums.LayerType;
import Tools.BufferUtil;
import Tools.Rand;
import org.lwjgl.BufferUtils;

public class CropLayer extends Layer {

    public Image getCropImage() {
        
        int h = outH;
        int w = outW;
        int c = outC;
        return new Image(w,h,c,output);
    }
    
    public CropLayer(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle, float saturation, float exposure) {
        
        this.type = LayerType.CROP;
        this.batch = batch;
        this.h = h;
        this.w = w;
        this.c = c;
        this.scale = (float)crop_height / h;
        this.flip = flip;
        this.angle = angle;
        this.saturation = saturation;
        this.exposure = exposure;
        this.outW = crop_width;
        this.outH = crop_height;
        this.outC = c;
        this.inputs = this.w * this.h * this.c;
        this.outputs = this.outW * this.outH * this.outC;

        this.output = BufferUtils.createFloatBuffer(this.outputs*batch);
    }

    public void resize(int width, int height) {

        w = width;
        h = height;

        outW =  (int) scale*width;
        outH =  (int) scale*height;

        inputs = w * h * c;
        outputs = outH * outW * outC;

        output = BufferUtil.reallocBuffer(output,batch*outputs);
    }

    public void forward(Network net) {

        int i,j,c,b,row,col;
        int index;
        int count = 0;

        int flip = (this.flip != 0 && Rand.randBoolean()) ? 1 : 0;

        int dh = Rand.randInt()%(this.h - this.outH + 1);
        int dw = Rand.randInt()%(this.w - this.outW + 1);
        float scale = 2;
        float trans = -1;

        if(this.noAdjust != 0){
            scale = 1;
            trans = 0;
        }

        if(net.train == 0){
            flip = 0;
            dh = (this.h - this.outH)/2;
            dw = (this.w - this.outW)/2;
        }

        for(b = 0; b < this.batch; ++b){
            for(c = 0; c < this.c; ++c){
                for(i = 0; i < this.outH; ++i){
                    for(j = 0; j < this.outW; ++j){
                        if(flip != 0){
                            col = this.w - dw - j - 1;
                        }
                        else{
                            col = j + dw;
                        }
                        row = i + dh;
                        index = col+this.w*(row+this.h*(c + this.c*b));

                        this.output.put(count++,net.input.get(index)*scale + trans);
                    }
                }
            }
        }
    }

    public void backward(Network net){}
}
