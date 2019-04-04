package Enums;

public enum CostType {

    SSE,
    MASKED,
    L1,
    SEG,
    SMOOTH,
    WGAN;

    public static CostType getCostType(String s) {
        switch(s.toUpperCase()) {
            case "SSE":
                return SSE;
            case "MASKED":
                return MASKED;
            case "L1":
                return L1;
            case "SEG":
                return SEG;
            case "SMOOTH":
                return SMOOTH;
            case "WGAN":
                return WGAN;
            default:
                return SSE;
        }
    }

    public static String getCostTypeString(CostType type) {
        return type.name();
    }
}
