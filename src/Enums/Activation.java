package Enums;

import Classes.Buffers.FloatBuffer;

public enum Activation {

    LOGISTIC,
    RELU,
    RELIE,
    LINEAR,
    RAMP,
    TANH,
    PLSE,
    LEAKY,
    ELU,
    LOGGY,
    STAIR,
    HARDTAN,
    LHTAN,
    SELU;

    public static Activation getActivation(String s) {

        String ss = s.toUpperCase();

        for (Activation a:Activation.values()) {
            if(a.name().equals(ss)) {
                return a;
            }
        }

        System.out.println("Activation::getActivation() -> Could not parse activation, using 'RELU'.");
        return RELU;
        }

    public static String getActivationString(Activation a) {

        return a.name().toLowerCase();
    }

    public static void activateArray(FloatBuffer x,final int n, Activation a) {

        for(int i = 0; i < n; ++i){

            float val = activate(x.get(i),a);
            x.put(i,val);
        }
    }

    public static void gradientArray(FloatBuffer x, int n, Activation a, FloatBuffer delta) {

        for(int i = 0; i < n; ++i){
            float val = delta.get(i) * gradient(x.get(i),a);
            delta.put(i,val);
        }
    }

    private static float activate(float x, Activation a) {

        switch(a){
            case LINEAR:
                return linearActivate(x);
            case LOGISTIC:
                return logisticActivate(x);
            case LOGGY:
                return loggyActivate(x);
            case RELU:
                return reluActivate(x);
            case ELU:
                return eluActivate(x);
            case SELU:
                return seluActivate(x);
            case RELIE:
                return relieActivate(x);
            case RAMP:
                return rampActivate(x);
            case LEAKY:
                return leakyActivate(x);
            case TANH:
                return tanhActivate(x);
            case PLSE:
                return plseActivate(x);
            case STAIR:
                return stairActivate(x);
            case HARDTAN:
                return hardtanActivate(x);
            case LHTAN:
                return lhtanActivate(x);
            default:
                return 0;
        }
    }

    private static float gradient(float x, Activation a) {

        switch(a){
            case LINEAR:
                return linearGradient(x);
            case LOGISTIC:
                return logisticGradient(x);
            case LOGGY:
                return loggyGradient(x);
            case RELU:
                return reluGradient(x);
            case ELU:
                return eluGradient(x);
            case SELU:
                return seluGradient(x);
            case RELIE:
                return relieGradient(x);
            case RAMP:
                return rampGradient(x);
            case LEAKY:
                return leakyGradient(x);
            case TANH:
                return tanhGradient(x);
            case PLSE:
                return plseGradient(x);
            case STAIR:
                return stairGradient(x);
            case HARDTAN:
                return hardtanGradient(x);
            case LHTAN:
                return lhtanGradient(x);
            default:
                return 0;
        }
    }

    private static float stairActivate(float x) {

        int n = (int) Math.floor(x);

        if (n%2 == 0)  {
            return (float) Math.floor(x/2.0);
        }
        else {
            return (x - n) + (float)Math.floor(x/2.0);
        }
    }

    private static float hardtanActivate(float x) {
        if (x < -1) return -1;
        if (x > 1) return 1;
        return x;
    }

    private static float linearActivate(float x){
        return x;
    }

    private static float logisticActivate(float x){
        return (float)(1.0/(1.0 + Math.exp(-x)));
    }

    private static float loggyActivate(float x){
        return (float)(2.0/(1.0 + Math.exp(-x)) - 1);
    }

    private static float reluActivate(float x){

        if(x > 0) {
            return x;
        }
        else {
            return 0;
        }
    }

    private static float eluActivate(float x){

        if(x >= 0) {
            return x;
        }
        else {
            return (float) Math.exp(x) - 1;
        }
    }

    private static float seluActivate(float x){

        if(x >= 0) {
            return 1.0507f*x;
        }
        else {
            return (float) (Math.exp(x) - 1) * 1.0507f * 1.6732f;
        }
    }

    private static float relieActivate(float x){
        return (x>0) ? x : 0.01f*x;
    }

    private static float rampActivate(float x){

        float val = 0.1f*x;

        if(x > 0) {
            val += x;
        }

        return val;
    }

    private static float leakyActivate(float x){

        if(x  <= 0) {
            x = x* 0.1f;
        }

        return x;
    }

    private static float tanhActivate(float x){
        return (float)((Math.exp(2*x)-1)/(Math.exp(2*x)+1));
    }

    private static float plseActivate(float x) {

        if(x < -4) {
            return 0.01f*(x + 4);
        }
        else if(x > 4) {
            return 0.01f * (x - 4) + 1;
        }
        else {
            return 0.5f + 0.125f*x;
        }
    }

    private static float lhtanActivate(float x) {

        if(x < 0) {
            return 0.001f*x;
        }
        else if(x > 1) {
            return 0.001f*(x - 1) + 1;
        }
        else {
            return x;
        }
    }

    private static float lhtanGradient(float x) {

        if(x > 0 && x < 1) {
            return 1;
        }
        else {
            return 0.001f;
        }
    }

    private static float hardtanGradient(float x) {

        if (x > -1 && x < 1)  {
            return 1;
        }
        else {
            return 0;
        }
    }

    private static float linearGradient(float x){
        return 1;
    }

    private static float logisticGradient(float x){
        return (1-x)*x;
    }

    private static float loggyGradient(float x) {

        float y = (x+1.0f)/2.0f;

        return 2*(1-y)*y;
    }

    private static float stairGradient(float x) {

        if (Math.floor(x) == x)  {
            return 0;
        }
        else {
            return 1;
        }
    }

    private static float reluGradient(float x){

        if(x > 0) {
            return 1;
        }
        else {
            return 0;
        }
    }

    private static float eluGradient(float x){

        if(x >= 0) {
            return 1;
        }
        else {
            return x + 1;
        }
    }

    private static float seluGradient(float x){

        if(x >= 0) {
            return 1.0507f;
        }
        else {
            return (x + 1.0507f * 1.6732f);
        }
    }

    private static float relieGradient(float x){

        return (x>0) ? 1 : 0.01f;
    }

    private static float rampGradient(float x){

        float val = 0.1f;

        if(x > 0) {
            val += 1;
        }

        return val;
    }

    private static float leakyGradient(float x){

        return (x>0) ? 1 : 0.1f;
    }

    private static float tanhGradient(float x){
        return 1-x*x;
    }

    private static float plseGradient(float x){
        return (x < 0 || x > 1) ? 0.01f : 0.125f;
    }
    
}
