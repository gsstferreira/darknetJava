package Tools;

import java.util.Random;

public abstract class Rand {

    public static final int MAX_INT = Integer.MAX_VALUE;
    public static final float MAX_FLOAT = Float.MAX_VALUE;

    private static Random rand = new Random();

    private static boolean haveSpare = false;
    private static double rand1;
    private static double rand2;

    public static float randUniform(float min, float max) {

        if(max < min) {
            return (rand.nextFloat() * (min - max)) + max;
        }
        else {
            return (rand.nextFloat() * (max - min)) + min;
        }
    }

    public static int randInt(int min, int max) {

        if(max < min) {
            return (rand.nextInt()%(min - max + 1)) + max;
        }
        else {
            return (rand.nextInt()%(max - min + 1)) + min;
        }
    }

    public static boolean randBoolean() {
        return rand.nextBoolean();
    }

    public static float randFloat() {

        return rand.nextFloat();
    }

    public static int randInt() {

        return rand.nextInt();
    }

    public static float randNormal() {

        double retVal;

        if(haveSpare)
        {
            retVal = Math.sqrt(rand1) * Math.sin(rand2);
        }
        else {

            rand1 =  -2 * Math.log(Math.max(1e-100,rand.nextDouble()));
            rand2 = 2 * Math.PI * rand.nextDouble();

            retVal = Math.sqrt(rand1) * Math.cos(rand2);
        }
        haveSpare = !haveSpare;

        return (float) retVal;
    }

    public static float randScale(float s) {

        float scale = randUniform(1, s);

        if(rand.nextBoolean()) {
            return scale;
        }
        else {
            return 1.0f / scale;
        }
    }

    public static void setRandSeed(long seed) {
        rand.setSeed(seed);
    }

}
