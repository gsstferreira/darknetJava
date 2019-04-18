package Yolo.Enums;

public enum LearningRatePolicy {

    CONSTANT,
    STEP,
    EXP,
    POLY,
    STEPS,
    SIG,
    RANDOM;

    public static LearningRatePolicy getLearningRatePolicy(String s) {
        switch (s.toUpperCase()) {
            case "CONSTANT":
                return CONSTANT;
            case "STEP":
                return  STEP;
            case "EXP":
                return EXP;
            case "POLY":
                return POLY;
            case "STEPS":
                return STEPS;
            case "SIG":
                return SIG;
            case "RANDOM":
                return RANDOM;
            default:
                return CONSTANT;
        }
    }
}
