package Classes;

import Classes.Buffers.FloatBuffer;
import Classes.Buffers.IntBuffer;
import Enums.ImType;
import Tools.Blas;
import Tools.GlobalVars;
import Tools.Rand;
import Tools.Util;
import org.lwjgl.BufferUtils;
import org.lwjgl.stb.STBImage;
import org.lwjgl.stb.STBImageWrite;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

public class Image {

    public int w;
    public int h;
    public int c;
    public FloatBuffer data;

    private static float[][] colors = {{1,0,1},{0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0}};

    private static final int alphabetNsize = 8;

    public Image(int width, int height, int c, boolean random) {
        this.w = width;
        this.h = height;
        this.c = c;

        this.data = new FloatBuffer(width*height*c);

        if(random) {
            for(int i = 0; i < w*h*c; ++i){

                double val = (Rand.randNormal() * 0.25) + 0.5;
                this.data.put(i,(float)val);
            }
        }
    }

    public Image(int width, int height, int c, FloatBuffer data) {

        this.w = width;
        this.h = height;
        this.c = c;

        this.data = data;

    }

    public static float getColor(int c, int x, int max) {

        float ratio = ((float)x/max)*5;
        int i = (int) Math.floor(ratio);
        int j = (int) Math.ceil(ratio);
        ratio -= i;
        float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
        return r;
    }

    public Image maskToRgb() {

        int n = this.c;
        Image im = new Image(w,h,3,false);
        int i, j;
        for(j = 0; j < n; ++j){
            int offset = j*123457 % n;
            float red = getColor(2,offset,n);
            float green = getColor(1,offset,n);
            float blue = getColor(0,offset,n);
            for(i = 0; i < im.w*im.h; ++i){

                int index = im.w*im.h;

                float val1 = im.data.get(i) + this.data.get(j*index + i) * red;
                float val2 = im.data.get(i + index) + this.data.get(j*index + i) * green;
                float val3 = im.data.get(i + 2*index) + this.data.get(j*index + i) * blue;

                im.data.put(i,val1);
                im.data.put(i + index, val2);
                im.data.put(i + 2*index, val3);
            }
        }
        return im;
    }

    public float getPixel(int x, int y, int c) {

        assert(x < this.w && y < this.h && c < this.c);

        return this.data.get(c*h*w + y*w + x);
    }

    public float getPixelExtended(int x, int y, int c) {

        if(x < 0 || x >= this.w || y < 0 || y >= this.h) {
            return 0;
        }

        else if(c < 0 || c >= this.c) {
            return 0;
        }
        else {
            return getPixel(x, y, c);
        }
    }

    public void setPixel(int x, int y, int c, float val) {

        if(x >= w || y >= h || c >= this.c || x < 0 || y < 0 || c < 0){
            return;
        }

        this.data.put(c*h*w + y*w + x,val);
    }

    public void addToPixel(int x, int y, int c, float val) {

        float v = getPixel(x,y,c);
        setPixel(x,y,c,val + v);
    }

    public float bilinearInterpolate(float x, float y, int c) {

        int ix = (int) Math.floor(x);
        int iy = (int) Math.floor(y);

        float dx = x - ix;
        float dy = y - iy;

        float val = (1-dy) * (1-dx) * getPixelExtended(ix,iy,c);
        val += dy * (1-dx) * getPixelExtended(ix,iy+1,c);
        val += (1-dy) * dx * getPixelExtended(ix+1,iy,c);
        val += dy * dx * getPixelExtended(ix+1,iy+1,c);

        return val;
    }

    public void compositeImage(Image dest, int dx, int dy) {

        int x,y,k;
        for(k = 0; k < c; ++k){
            for(y = 0; y < h; ++y){
                for(x = 0; x < w; ++x){
                    float val = this.getPixel(x,y,k);
                    float val2 = dest.getPixelExtended(dx+x, dy+y, k);
                    dest.setPixel(dx+x, dy+y, k, val * val2);
                }
            }
        }
    }

    public Image borderImage(int border) {

        Image b = new Image(w + 2*border, h + 2*border, c,false);
        int x,y,k;
        for(k = 0; k < b.c; ++k){
            for(y = 0; y < b.h; ++y){
                for(x = 0; x < b.w; ++x){
                    float val = this.getPixelExtended(x - border, y - border, k);
                    if((x - border) < 0 || (x - border) >= w || (y - border) < 0 || (y - border) >= h) {
                        val = 1;
                    }
                    b.setPixel(x, y, k, val);
                }
            }
        }
        return b;
    }

    public Image tileImages(Image b, int dx) {

        if(w == 0) {
            return b.copyImage();
        }
        else {
            Image c = new Image(w + b.w + dx,(h > b.h) ? h : b.h, (this.c > b.c) ? this.c : b.c,false);
            Blas.fillCpu(c.w*c.h*c.c, 1, c.data, 1);
            this.embedImage(c, 0, 0);
            b.compositeImage(c, w + dx, 0);
            return c;
        }
    }

    public Image getLabel(Image[][] characters, String string, int size) {

        size = size/10;
        if(size > 7) size = 7;
        Image label = new Image(0,0,0,false);

        for (int i = 0; i < string.length(); i++) {

            Image l = characters[size][(0x0000FFFF & string.charAt(i))];
            label = label.tileImages(l,-size - 1 + (size + 1)/2);
        }
        return label.borderImage((int) (label.h*0.25f));
    }

    public void embedImage(Image dest, int dx, int dy) {

        int x,y,k;
        for(k = 0; k < c; ++k){
            for(y = 0; y < h; ++y){
                for(x = 0; x < w; ++x){
                    float val = getPixel(x,y,k);
                    dest.setPixel(dx+x, dy+y, k, val);
                }
            }
        }
    }

    public Image copyImage() {

        Image copy = new Image(w,h,c,false);

        for(int i = 0; i < data.size(); i++) {
            copy.data.put(i,data.get(i));
        }
        return copy;
    }

    public Image resizeImage(int w, int h) {

        Image resized = new Image(w, h, this.c,false);
        Image part = new Image(w, this.h, this.c,false);

        int r, c, k;
        float w_scale = (float)(this.w - 1) / (w - 1);
        float h_scale = (float)(this.h - 1) / (h - 1);
        for(k = 0; k < this.c; ++k){
            for(r = 0; r < this.h; ++r){
                for(c = 0; c < w; ++c){
                    float val = 0;
                    if(c == w-1 || this.w == 1){
                        val = getPixel(this.w-1, r, k);
                    } else {
                        float sx = c*w_scale;
                        int ix = (int) sx;
                        float dx = sx - ix;
                        val = (1 - dx) * getPixel(ix, r, k) + dx * getPixel(ix+1, r, k);
                    }
                    part.setPixel(c, r, k, val);
                }
            }
        }
        for(k = 0; k < this.c; ++k){
            for(r = 0; r < h; ++r){
                float sy = r*h_scale;
                int iy = (int) sy;
                float dy = sy - iy;
                for(c = 0; c < w; ++c){
                    float val = (1-dy) * part.getPixel(c, iy, k);
                    resized.setPixel(c, r, k, val);
                }
                if(r == h-1 || this.h == 1) continue;
                for(c = 0; c < w; ++c){
                    float val = dy * part.getPixel(c, iy + 1, k);
                    resized.addToPixel(c, r, k, val);
                }
            }
        }
        return resized;
    }

    public void drawLabel(int r, int c, Image label, FloatBuffer rgb) {

        int _w = label.w;
        int _h = label.h;
        if ((r - _h) >= 0) {
            r -= _h;
        }

        int i, j, k;
        for(j = 0; (j < _h) && ((j + r) < h); ++j){
            for(i = 0; (i < _w) && ((i + c) < w); ++i){
                for(k = 0; k < label.c; ++k){
                    float val = label.getPixel(i, j, k);
                    setPixel(i+c, j+r, k, rgb.get(k) * val);
                }
            }
        }
    }

    public void drawBox(int x1, int y1, int x2, int y2, float r, float g, float b) {

        int i;
        if(x1 < 0) x1 = 0;
        else if(x1 >= w) x1 = w-1;
        if(x2 < 0) x2 = 0;
        else if(x2 >= w) x2 = w-1;

        if(y1 < 0) y1 = 0;
        else if(y1 >= h) y1 = h-1;
        if(y2 < 0) y2 = 0;
        else if(y2 >= h) y2 = h-1;

        for(i = x1; i <= x2; ++i) {

            this.data.put(i + y1*w,r);
            this.data.put(i + y2*w,r);

            this.data.put(i + y1*w + w*h,g);
            this.data.put(i + y2*w + w*h,g);

            this.data.put(i + y1*w + 2*h*w,b);
            this.data.put(i + y2*w + 2*h*w,b);
        }
        for(i = y1; i <= y2; ++i) {

            this.data.put(x1 + i*w,r);
            this.data.put(x2 + i*w,r);

            this.data.put(x1 + i*w + w*h,g);
            this.data.put(x2 + i*w + w*h,g);

            this.data.put(x1 + i*w + 2*w*h,b);
            this.data.put(x2 + i*w + 2*w*h,b);
        }
    }

    public void drawBoxWidth(int x1, int y1, int x2, int y2, int w, float r, float g, float b) {

        int i;
        for(i = 0; i < w; ++i){
            drawBox(x1+i, y1+i, x2-i, y2-i, r, g, b);
        }
    }

    public void drawBbox(Box bbox, int w, float r, float g, float b) {

        int left  = (int) (bbox.x-bbox.w/2)*this.w;
        int right = (int) (bbox.x+bbox.w/2)*this.w;
        int top   = (int) (bbox.y-bbox.h/2)*this.h;
        int bot   = (int) (bbox.y+bbox.h/2)*this.h;

        int i;
        for(i = 0; i < w; ++i){
            drawBox(left+i, top+i, right-i, bot-i, r, g, b);
        }
    }

    public static Image loadImageColor(String fileName,int w, int h) {
        return loadImage(fileName,w,h,3);
    }

    public static Image loadImage(String filename, int w, int h, int c) {

        Image out = loadImageStb(filename, c);

        if((h != 0 && w != 0) && (h != out.h || w != out.w)){

            out = out.resizeImage(w, h);
        }
        return out;
    }

    public static Image loadImageStb(String filename, int channels) {

        //Emulando ponteiros
        int[] _width = new int[1];
        int[] _height = new int[1];
        int[] _channel = new int[1];

        try {
            ByteBuffer bb;

            try {
                InputStream inputStream = Image.class.getResourceAsStream(filename);
                byte[] b = inputStream.readAllBytes();
                inputStream.close();
                ByteBuffer _BB = BufferUtils.createByteBuffer(b.length);
                _BB.put(b);
                _BB.position(0);
                bb = STBImage.stbi_load_from_memory(_BB,_width,_height,_channel,channels);

            }
            catch (Exception e){
                bb = STBImage.stbi_load(filename,_width,_height,_channel,channels);
            }

            int width = _width[0];
            int height = _height[0];
            int channel = (channels != 0) ? channels : _channel[0];

            Image im = new Image(width, height, channels,false);

            for(int k = 0; k < channel; ++k){
                for(int j = 0; j < height; ++j){
                    for(int i = 0; i < width; ++i){
                        int dst_index = i + width*j + width*height*k;
                        int src_index = k + channel*i + channel*width*j;

                        float val = (bb.get(src_index) & 0x000000FF)/255.0f;
                        im.data.put(dst_index,val);
                    }
                }
            }
            return im;
        }
        catch (Exception e) {
            System.out.println(String.format("Cannot load Image '%s'",filename));
            e.printStackTrace();
            System.exit(-1);
            return null;
        }
    }

    private void saveImageOptions(String name, ImType f, int quality) {

        String s;

        switch (f) {
            case BMP:
                s = String.format("%s.bmp",name);
                break;
            case JPG:
                s = String.format("%s.jpg",name);
                break;
            case TGA:
                s = String.format("%s.tga",name);
                break;
            default:
                s = String.format("%s.png",name);
                break;
        }

        byte[] data = new byte[w*h*c];

        int i,k;
        for(k = 0; k < c; ++k){
            for(i = 0; i < w*h; ++i){
                data[i*c+k] = (byte)(255.0f*this.data.get(i+k*w*h));
            }
        }

        ByteBuffer bb = BufferUtils.createByteBuffer(data.length);
        bb = bb.put(data,0,data.length);
        bb.position(0);

        boolean success;

        switch (f) {
            case BMP:
                success = STBImageWrite.stbi_write_bmp(s,w,h,c,bb);
                break;
            case JPG:
                success = STBImageWrite.stbi_write_jpg(s,w,h,c,bb,quality);
                break;
            case TGA:
                success = STBImageWrite.stbi_write_tga(s,w,h,c,bb);
                break;
            default:
                success = STBImageWrite.stbi_write_png(s,w,h,c,bb,w*c);
                break;
        }
        if(!success) {
            System.out.println(String.format("Cannot save Image Image '%s'",s));
        }
    }

    public void saveToDisk(String path, ImType f, int quality) {

        this.saveImageOptions(path,f,quality);
    }

    public void saveToDisk(String path, ImType f) {

        this.saveImageOptions(path,f,100);
    }

    public static Image[][] loadAlphabet() {

        Image[][] alphabets = new Image[alphabetNsize][128];

        IntStream.range(0,alphabetNsize).parallel().forEach(j -> {
            for(int i = 32; i < 127; ++i){
                String s = String.format("/Res/labels/%d_%d.png",i,j);
                alphabets[j][i] = loadImageColor(s, 0, 0);
            }
        });
        return alphabets;
    }

    public Image getImageLayer(int l) {
        Image out = new Image(w,h,1,false);

        for(int i = 0; i < w*h; i++) {
            out.data.put(i,this.data.get(i+l*h*w));
        }
        return out;
    }

    public static Image collapseImagesVert(Image[] ims, int n) {

        int border = 1;
        int w = ims[0].w;
        int h = (ims[0].h + border) * n - border;
        int c = ims[0].c;

        if(c != 3) {
            w = (w+border)*c - border;
            c = 1;
        }

        Image filters = new Image(w,h,c,false);

        for(int i = 0; i < n; ++i){
            int h_offset = i*(ims[0].h+border);
            Image copy = ims[i].copyImage();

            if(c == 3){
                copy.embedImage(filters, 0, h_offset);
            }
            else{
                for(int j = 0; j < copy.c; ++j){
                    int w_offset = j*(ims[0].w+border);
                    Image layer = copy.getImageLayer(j);
                    layer.embedImage(filters, w_offset, h_offset);
                }
            }
        }
        return filters;
    }

    public static Image collapseImagesHorz(Image[] ims, int n) {


        int color = 1;
        int border = 1;

        int size = ims[0].h;
        int h = size;
        int w = (ims[0].w + border) * n - border;
        int c = ims[0].c;

        if(c != 3){
            h = (h+border)*c - border;
            c = 1;
        }

        Image filters = new Image(w, h, c,false);

        for(int i = 0; i < n; ++i){
            int w_offset = i*(size+border);
            Image copy = ims[i].copyImage();

            if(c == 3){
                copy.embedImage(filters, w_offset, 0);
            }
            else{
                for(int j = 0; j < copy.c; ++j){
                    int h_offset = j*(size+border);
                    Image layer = copy.getImageLayer(j);
                    layer.embedImage(filters, w_offset, h_offset);
                }
            }
        }
        return filters;
    }

    public void transpose() {

        assert(w == h);
        for(int i = 0; i < c; ++i){
            for(int n = 0; n < w-1; ++n){
                for(int m = n + 1; m < w; ++m){

                    int indexN = n + w*(m + h*i);
                    int indexM = m + w*(n + h*i);

                    float swap = data.get(indexM);

                    data.put(indexM,data.get(indexN));
                    data.put(indexN,swap);
                }
            }
        }
    }

    public void rotateCw(int times) {

        assert(w == h);
        times = (times + 400) % 4;
        int i, x, y, z;
        int n = w;
        for(i = 0; i < times; ++i){
            for(z = 0; z < c; ++z){
                for(x = 0; x < n/2; ++x){
                    for(y = 0; y < (n-1)/2 + 1; ++y){

                        float temp = data.get(y + w*(x + h*z));
                        data.put(y + w*(x + h*z),n-1-x + w*(y + h*z));
                        data.put(n-1-x + w*(y + h*z),n-1-y + w*(n-1-x + h*z));
                        data.put(n-1-y + w*(n-1-x + h*z),x + w*(n-1-y + h*z));
                        data.put(x + w*(n-1-y + h*z),temp);
                    }
                }
            }
        }
    }

    public void flip() {

        for(int k = 0; k < c; ++k){
            for(int i = 0; i < h; ++i){
                for(int j = 0; j < w/2; ++j){

                    int index = j + w*(i + h*(k));
                    int flip = (w - j - 1) + w*(i + h*(k));
                    float swap = data.get(flip);
                    data.put(flip,index);
                    data.put(index,swap);
                }
            }
        }
    }

    public static Image ImageDistance(Image a, Image b) {

        Image dist = new Image(a.w, a.h, 1,false);
        for(int i = 0; i < a.c; ++i){
            for(int j = 0; j < a.h*a.w; ++j){

                float val = dist.data.get(j) + (float)Math.pow(a.data.get(i*a.h*a.w+j) - b.data.get(i*a.h*a.w+j),2);
                dist.data.put(j,val);
            }
        }
        for(int j = 0; j < a.h*a.w; ++j){

            dist.data.put(j,(float)Math.sqrt(dist.data.get(j)));
        }
        return dist;
    }

    public static void ghostImage(Image source, Image dest, int dx, int dy) {

        float max_dist = (float)Math.sqrt((-source.w/2. + .5)*(-source.w/2. + .5));
        for(int k = 0; k < source.c; ++k){
            for(int y = 0; y < source.h; ++y){
                for(int x = 0; x < source.w; ++x){
                    float dist = (float)Math.sqrt((x - source.w/2. + .5)*(x - source.w/2. + .5) + (y - source.h/2. + .5)*(y - source.h/2. + .5));
                    float alpha = (1 - dist/max_dist);
                    if(alpha < 0) alpha = 0;
                    float v1 = source.getPixel(x,y,k);
                    float v2 = dest.getPixel(dx+x,dy+y,k);
                    float val = alpha*v1 + (1-alpha)*v2;
                    dest.setPixel(dx+x, dy+y, k, val);
                }
            }
        }
    }

    public void blocky(int s) {

        for(int k = 0; k < c; ++k){
            for(int j = 0; j < h; ++j){
                for(int i = 0; i < w; ++i){
                    data.put(i + w*(j + h*k),data.get(i/s*s + w*(j/s*s + h*k)));
                }
            }
        }
    }

    public void censor(int dx, int dy, int w, int h) {

        int s = 32;
        if(dx < 0) dx = 0;
        if(dy < 0) dy = 0;

        for(int k = 0; k < c; ++k){
            for(int j = dy; j < dy + h && j < h; ++j){
                for(int i = dx; i < dx + w && i < w; ++i){

                    data.put(i + w*(j + h*k),data.get(i/s*s + w*(j/s*s + h*k)));
                }
            }
        }
    }

    public void constrain() {

        for(int i = 0; i < w*h*c; ++i){
            if(data.get(i) < 0) {
                data.put(i,0);
            }
            else if(data.get(i) > 1) {
                data.put(i,1);
            }
        }
    }

    public void normalize() {

        float min = 9999999;
        float max = -999999;

        for(int i = 0; i < h*w*c; ++i){
            float v = data.get(i);
            if(v < min) {
                min = v;
            }
            else if(v > max) {
                max = v;
            }
        }
        if(max - min < .000000001){
            min = 0;
            max = 1;
        }
        for(int i = 0; i < c*w*h; ++i){

            data.put(i,(data.get(i) - min)/(max-min));
        }
    }

    public void normalize2() {

        float[] min = new float[c];
        float[] max = new float[c];

        for(int i = 0; i < c; ++i) {
            min[i] = data.get(i*h*w);
            max[i] = data.get(i*h*w);
        }

        for(int j = 0; j < c; ++j){
            for(int i = 0; i < h*w; ++i){

                float v = data.get(i+j*h*w);

                if(v < min[j]) {
                    min[j] = v;
                }
                if(v > max[j]) {
                    max[j] = v;
                }
            }
        }

        for(int i = 0; i < c; ++i){
            if(max[i] - min[i] < .000000001){
                min[i] = 0;
                max[i] = 1;
            }
        }

        for(int j = 0; j < c; ++j){
            for(int i = 0; i < w*h; ++i){
                data.put(i+j*h*w,(data.get(i+j*h*w) - min[j])/(max[j]-min[j]));
            }
        }
    }

    public void rgbgr() {

        int i;
        for(i = 0; i < w*h; ++i){
            float swap = data.get(i);
            data.put(i,i+w*h*2);
            data.put(i+w*h*2,swap);
        }
    }

    public void placeImage(int w, int h, int dx, int dy, Image canvas) {
        
        int x, y, c;
        for(c = 0; c < c; ++c){
            for(y = 0; y < h; ++y){
                for(x = 0; x < w; ++x){
                    
                    float rx = ((float)x / w) * w;
                    float ry = ((float)y / h) * h;
                    float val = bilinearInterpolate(rx, ry, c);
                    canvas.setPixel(x + dx, y + dy, c, val);
                }
            }
        }
    }

    public Image centerCrop(int w, int h) {

        int m = (w < h) ? w : h;
        Image c =  crop((w - m) / 2, (h - m)/2, m, m);
        Image r = c.resizeImage(w, h);
        return r;
    }

    public Image rotateCrop(float rad, float s, int w, int h, float dx, float dy, float aspect) {

        float cx = w/2.0f;
        float cy = h/2.0f;

        Image rot = new Image(w, h, c,false);
        for(int z = 0; z < c; ++z){
            for(int y = 0; y < h; ++y){
                for(int x = 0; x < w; ++x){

                    double rx = Math.cos(rad)*((x - w/2.)/s*aspect + dx/s*aspect) - Math.sin(rad)*((y - h/2.)/s + dy/s) + cx;
                    double ry = Math.sin(rad)*((x - w/2.)/s*aspect + dx/s*aspect) + Math.cos(rad)*((y - h/2.)/s + dy/s) + cy;

                    float val = bilinearInterpolate((float)rx, (float)ry, z);
                    rot.setPixel(x, y, z, val);
                }
            }
        }
        return rot;
    }

    public Image rotate(float rad) {

        float cx = w/2.0f;
        float cy = h/2.0f;
        Image rot = new Image(w, h, c,false);
        for(int z = 0; z < c; ++z){
            for(int y = 0; y < h; ++y){
                for(int x = 0; x < w; ++x){
                    double rx = Math.cos(rad)*(x-cx) - Math.sin(rad)*(y-cy) + cx;
                    double ry = Math.sin(rad)*(x-cx) + Math.cos(rad)*(y-cy) + cy;
                    float val = bilinearInterpolate((float)rx, (float)ry, z);
                    rot.setPixel(x, y, z, val);
                }
            }
        }
        return rot;
    }

    public void fill(float s) {

        int i;
        for(i = 0; i < h*w*c; ++i) {

            data.put(i,s);
        }
    }

    public void translate(float s) {

        for(int i = 0; i < h*w*c; ++i) {

            data.put(i,data.get(i) + s);
        }
    }

    public void scale(float s) {

        for(int i = 0; i < h*w*c; ++i) {

            data.put(i,data.get(i) * s);
        }
    }

    public Image crop(int dx, int dy, int width, int height) {

        Image cropped = new Image(width, height, c,false);

        for(int k = 0; k < c; ++k){
            for(int j = 0; j < height; ++j){
                for(int i = 0; i < width; ++i){
                    int r = j + dy;
                    int c = i + dx;
                    float val = 0;
                    r = Util.constrain(r, 0, h-1);
                    c = Util.constrain(c, 0, w-1);
                    val = getPixel(c, r, k);
                    cropped.setPixel(i, j, k, val);
                }
            }
        }
        return cropped;
    }
    
    public static int best3DShiftR(Image a, Image b, int min, int max) {

        if(min == max) return min;
        int mid = (int) Math.floor((min + max) / 2.);
        Image c1 = b.crop(0, mid, b.w, b.h);
        Image c2 = b.crop(0, mid+1, b.w, b.h);

        float d1 = Util.distArray(c1.data, a.data, a.w*a.h*a.c, 10);
        float d2 = Util.distArray(c2.data, a.data, a.w*a.h*a.c, 10);

        if(d1 < d2) {
            return best3DShiftR(a, b, min, mid);
        }
        else {
            return best3DShiftR(a, b, mid+1, max);
        }
    }

    public static int best3DShift(Image a, Image b, int min, int max) {

        int i;
        int best = 0;
        float best_distance = Rand.MAX_FLOAT;
        for(i = min; i <= max; i += 2){
            Image c = b.crop(0, i, b.w, b.h);
            float d = Util.distArray(c.data, a.data, a.w*a.h*a.c, 100);
            if(d < best_distance){
                best_distance = d;
                best = i;
            }
        }
        return best;
    }

    public void composite3D(String f1, String f2, String out, int delta) {

        if(out == null || out.isEmpty())  {
            out = "out";
        }
        Image a = Image.loadImage(f1, 0,0,0);
        Image b = Image.loadImage(f2, 0,0,0);
        int shift = best3DShiftR(a, b, -a.h/100, a.h/100);

        Image c1 = b.crop(10, shift, b.w, b.h);
        float d1 = Util.distArray(c1.data, a.data, a.w*a.h*a.c, 100);
        Image c2 = b.crop(-10, shift, b.w, b.h);
        float d2 = Util.distArray(c2.data, a.data, a.w*a.h*a.c, 100);

        Image c = b.crop(delta, shift, a.w, a.h);
        int i;
        for(i = 0; i < c.w*c.h; ++i){

            c.data.put(i,a.data.get(i));
        }
        c.saveToDisk(out,ImType.JPG,80);
    }

    public void letterboxImageInto(int width, int height, Image boxed) {

        int new_w;
        int new_h;

        if ((1.0f*width/w) < (1.0f*height/h)) {
            new_w = width;
            new_h = (h * width)/w;
        }
        else {
            new_h = height;
            new_w = (w * height)/h;
        }
        Image resized = resizeImage(new_w, new_h);

        resized.embedImage(boxed, (width-new_w)/2, (height-new_h)/2);
    }

    public Image letterbox(int width, int height) {

        int new_w;
        int new_h;

        if ((1.0f*width/w) < (1.0f*height/h)) {
            new_w = width;
            new_h = (h * width)/w;
        }
        else {
            new_h = height;
            new_w = (w * height)/h;
        }
        Image resized = resizeImage(new_w, new_h);
        Image boxed = new Image(width, height, c,false);
        boxed.fill( 0.5f);

        resized.embedImage(boxed, (width-new_w)/2, (height-new_h)/2);
        return boxed;
    }

    public Image resizeMax(int max) {

        int width = w;
        int height = h;

        if(width > height){
            height = (height * max) / width;
            width = max;
        }
        else {
            width = (width * max) / height;
            height = max;
        }

        if(width == w && height == h) {
            return this;
        }
        else {
            return resizeImage(width, height);
        }
    }

    public Image resizeMin(int min) {

        int width = w;
        int height = h;

        if(width < height){
            height = (height * min) / width;
            width = min;
        }
        else {
            width = (width * min) / height;
            height = min;
        }

        if(width == w && height == h) {
            return this;
        }
        return resizeImage(width,height);
    }

    public Image randomCrop(int width, int height) {

        int dx = Rand.randInt(0,w - width);
        int dy = Rand.randInt(0,h - height);

        return crop(dx,dy,width,height);
    }
    
    public AugmentArgs randomAugmentArgs(float angle, float aspect, int low, int high, int width, int height) {
        
        AugmentArgs a = new AugmentArgs();
        
        aspect = Rand.randScale(aspect);
        int r = Rand.randInt(low, high);
        int min = (int) ((h < w*aspect) ? h : w*aspect);
        float scale = (float)r / min;

        float rad = Rand.randUniform(-angle, angle) * 2 * (float)Math.PI / 360.0f;

        float dx = (w*scale/aspect - width) / 2.0f;
        float dy = (h*scale - width) / 2.0f;

        
        dx = Rand.randUniform(-dx, dx);
        dy = Rand.randUniform(-dy, dy);

        a.rad = rad;
        a.scale = scale;
        a.w = width;
        a.h = height;
        a.dx = dx;
        a.dy = dy;
        a.aspect = aspect;
        return a;
    }

    public Image randomAugment(float angle, float aspect, int low, int high, int w, int h) {

        AugmentArgs a = randomAugmentArgs(angle, aspect, low, high, w, h);
        return rotateCrop(a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);
    }

    private static float threeWayMax(float a, float b, float c) {

        return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
    }

    private static float threeWayMin(float a, float b, float c) {

        return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
    }

    public void yuvToRgb() {

        assert(c == 3);
        float r, g, b;
        float y, u, v;
        for(int j = 0; j < h; ++j){
            for(int i = 0; i < w; ++i){
                y = getPixel(i , j, 0);
                u = getPixel(i , j, 1);
                v = getPixel(i , j, 2);

                r = y + 1.13983f*v;
                g = y + -.39465f*u + -.58060f*v;
                b = y + 2.03211f*u;

                setPixel(i, j, 0, r);
                setPixel(i, j, 1, g);
                setPixel(i, j, 2, b);
            }
        }
    }

    public void rgbToYuv() {

        assert(c == 3);
        int i, j;
        float r, g, b;
        float y, u, v;
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                r = getPixel(i , j, 0);
                g = getPixel(i , j, 1);
                b = getPixel(i , j, 2);

                y = .299f*r + .587f*g + .114f*b;
                u = -.14713f*r + -.28886f*g + .436f*b;
                v = .615f*r + -.51499f*g + -.10001f*b;

                setPixel(i, j, 0, y);
                setPixel(i, j, 1, u);
                setPixel(i, j, 2, v);
            }
        }
    }

    public void rgbToHsv() {

        assert(c == 3);
        int i, j;
        float r, g, b;
        float h, s, v;
        for(j = 0; j < this.h; ++j){
            for(i = 0; i < w; ++i){
                r = getPixel(i , j, 0);
                g = getPixel(i , j, 1);
                b = getPixel(i , j, 2);
                float max = threeWayMax(r,g,b);
                float min = threeWayMin(r,g,b);
                float delta = max - min;
                v = max;
                if(max == 0){
                    s = 0;
                    h = 0;
                }
                else{
                    s = delta/max;
                    if(r == max){
                        h = (g - b) / delta;
                    }
                    else if (g == max) {
                        h = 2 + (b - r) / delta;
                    }
                    else {
                        h = 4 + (r - g) / delta;
                    }

                    if (h < 0) h += 6;
                    h = h/6.0f;
                }
                setPixel(i, j, 0, h);
                setPixel(i, j, 1, s);
                setPixel(i, j, 2, v);
            }
        }
    }

    public void hsvToRgb() {

        assert(c == 3);
        int i, j;
        float r, g, b;
        float h, s, v;
        float f, p, q, t;
        for(j = 0; j < this.h; ++j){
            for(i = 0; i < w; ++i){
                h = 6 * getPixel(i , j, 0);
                s = getPixel(i , j, 1);
                v = getPixel(i , j, 2);

                if (s == 0) {
                    r = g = b = v;
                }
                else {
                    int index = (int) Math.floor(h);
                    f = h - index;
                    p = v*(1-s);
                    q = v*(1-s*f);
                    t = v*(1-s*(1-f));

                    if(index == 0){
                        r = v; g = t; b = p;
                    }
                    else if(index == 1){
                        r = q; g = v; b = p;
                    }
                    else if(index == 2){
                        r = p; g = v; b = t;
                    }
                    else if(index == 3){
                        r = p; g = q; b = v;
                    }
                    else if(index == 4){
                        r = t; g = p; b = v;
                    }
                    else {
                        r = v; g = p; b = q;
                    }
                }
                setPixel(i, j, 0, r);
                setPixel(i, j, 1, g);
                setPixel(i, j, 2, b);
            }
        }
    }
    
    public void grayscaleImage3C() {
        
        assert(c == 3);
        int i, j, k;
        float[] scale = {0.299f, 0.587f, 0.114f};
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                float val = 0;
                for(k = 0; k < 3; ++k){
                    val += scale[k]*getPixel(i, j, k);
                }
                
                data.put(w*j + i,val);
                data.put(h*w + w*j + i,val);
                data.put(2*h*w + w*j + i,val);
            }
        }
    }

    public Image grayscaleImage() {

        assert(c == 3);
        int i, j, k;
        Image gray = new Image(w, h, 1,false);
        float[] scale = {0.299f, 0.587f, 0.114f};
        for(k = 0; k < c; ++k){
            for(j = 0; j < h; ++j){
                for(i = 0; i < w; ++i){

                    float val = data.get(i + w*j) + scale[k]*getPixel( i, j, k);
                    data.put(i + w*j,val);
                }
            }
        }
        return gray;
    }

    public Image thresholdImage(float thresh) {

        Image t = new Image(w, h, c, false);
        for(int i = 0; i < w*h*c; ++i){

            t.data.put(i,(data.get(i) > thresh) ? 1 : 0);
        }
        return t;
    }

    public static Image blendImage(Image fore, Image back, float alpha) {

        assert(fore.w == back.w && fore.h == back.h && fore.c == back.c);
        Image blend = new Image(fore.w, fore.h, fore.c,false);
        int i, j, k;
        for(k = 0; k < fore.c; ++k){
            for(j = 0; j < fore.h; ++j){
                for(i = 0; i < fore.w; ++i){
                    float val = alpha * fore.getPixel(i, j, k) + (1 - alpha)* back.getPixel(i, j, k);
                    blend.setPixel(i, j, k, val);
                }
            }
        }
        return blend;
    }

    public void scaleImageChannel(int c, float v) {

        int i, j;
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                float pix = getPixel(i, j, c);
                pix = pix*v;
                setPixel(i, j, c, pix);
            }
        }
    }

    public void translateImageChannel(int c, float v) {

        for(int j = 0; j < h; ++j){
            for(int i = 0; i < w; ++i){
                float pix = getPixel(i, j, c);
                pix = pix+v;
                setPixel(i, j, c, pix);
            }
        }
    }

    public Image binarize() {

        Image bin = copyImage();
        int i;
        for(i = 0; i < w * h * c; ++i){
            if(bin.data.get(i) > .5) {
                bin.data.put(i,1);
            }
            else {
                bin.data.put(i,0);
            }
        }
        return bin;
    }

    public void saturate(float sat) {

        rgbToHsv();
        scaleImageChannel(1,sat);
        hsvToRgb();
        constrain();
    }

    public void hue(float hue) {

        rgbToHsv();

        for(int i = 0; i < w*h; ++i){

            data.put(i,data.get(i) + hue);

            if(data.get(i) > 1) {
                data.put(i,data.get(i) - 1);
            }
            else if(data.get(i) < 0) {
                data.put(i,data.get(i) + 1);
            }
        }
        hsvToRgb();
        constrain();
    }

    public void exposure(float sat) {

        rgbToHsv();
        scaleImageChannel(2,sat);
        hsvToRgb();
        constrain();
    }

    public void distort(float hue, float sat, float val) {

        rgbToHsv();
        scaleImageChannel(1,sat);
        scaleImageChannel(2,val);

        for(int i = 0; i < w*h; ++i){

            data.put(i,data.get(i) + hue);

            if(data.get(i) > 1) {
                data.put(i,data.get(i) - 1);
            }
            else if(data.get(i) < 0) {
                data.put(i,data.get(i) + 1);
            }
        }
        hsvToRgb();
        constrain();
    }

    public void randomDistort(float hue, float saturation, float exposure) {

        float dhue = Rand.randUniform(-hue, hue);
        float dsat = Rand.randScale(saturation);
        float dexp = Rand.randScale(exposure);
        distort(dhue, dsat, dexp);
    }

    public void saturateExposure(float sat, float exposure) {

        rgbToHsv();
        scaleImageChannel(1,sat);
        scaleImageChannel(2,exposure);
        hsvToRgb();
        constrain();
    }

    public List<Result> drawDetections(Detection[] dets, int num, float thresh, List<String> names, Image[][] alphabet, int classes) {

        int i,j;
        List<Result> list = new ArrayList<>();

        //System.out.print("[");

        for(i = 0; i < num; ++i){

            String labelString = "";
            String currentName = "";
            float currentConfidence = 0;

            int _class = -1;
            for(j = 0; j < classes; ++j){
                if (dets[i].prob[j] > thresh){
                    if (_class < 0) {

                        labelString = labelString.concat(names.get(j));
                        _class = j;
                    }
                    else {

                        labelString = labelString.concat(", ");
                        labelString = labelString.concat(names.get(j));
                    }
                    currentName = names.get(j);
                    currentConfidence = dets[i].prob[j] * 100.0f;
                    //System.out.print(String.format("(%s:%.0f)",currentName,currentConfidence));
                }
            }

            if(_class >= 0){

                int width = (int) (h * 0.006f);
                
                int offset = _class*123457 % classes;
                float red = getColor(2,offset,classes);
                float green = getColor(1,offset,classes);
                float blue = getColor(0,offset,classes);
                float[] rgb = new float[3];

                rgb[0] = red;
                rgb[1] = green;
                rgb[2] = blue;

                Box b = dets[i].bBox;

                int left  = (int)((b.x-b.w/2.)*w);
                int right = (int)((b.x+b.w/2.)*w);
                int top   = (int)((b.y-b.h/2.)*h);
                int bot   = (int)((b.y+b.h/2.)*h);

                if(left < 0) {
                    left = 0;
                }
                if(right > w - 1) {
                    right = w - 1;
                }
                if(top < 0) {
                    top = 0;
                }
                if(bot > h - 1) {
                    bot = h - 1;
                }

                Result r = new Result(currentName,currentConfidence,left,top,right-left,bot-top);
                list.add(r);

                drawBoxWidth(left, top, right, bot, width, red, green, blue);

                if (alphabet != null) {
                    Image label = getLabel(alphabet, labelString, (int)(h*.03f));
                    drawLabel(top + width, left, label, new FloatBuffer(rgb));
                }

                if (dets[i].mask != null) {

                    Image mask = new Image(14, 14, 1, new FloatBuffer(dets[i].mask));

                    Image resized_mask = mask.resizeImage((int) b.w*w, (int) b.h*h);
                    Image tmask = resized_mask.thresholdImage(0.5f);

                    tmask.embedImage(this,left,top);
                }
            }
        }
        //System.out.println("]");
        return list;
    }

}
