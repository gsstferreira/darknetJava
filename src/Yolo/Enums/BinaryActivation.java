package Yolo.Enums;

public enum BinaryActivation {

    MULT,
    ADD,
    SUB,
    DIV;

    public static BinaryActivation getBinaryAcgtivation(String s) {

        switch(s.toUpperCase()) {
            case "MULT":
                return MULT;
            case "ADD":
                return ADD;
            case "SUB":
                return SUB;
            case "DIV":
                return DIV;
            default:
                return ADD;
        }
    }
}
