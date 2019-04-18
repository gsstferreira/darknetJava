package Yolo.Enums;

public enum DataType {

    CLASSIFICATION,
    DETECTION,
    CAPTCHA,
    REGION,
    IMAGE,
    COMPARE,
    WRITING,
    SWAG,
    TAG,
    OLD_CLASSIFICATION,
    STUDY,
    DET,
    SUPER,
    LETTERBOX,
    REGRESSION,
    SEGMENTATION,
    INSTANCE,
    ISEG;

    public static DataType getDataType(String s) {

        switch (s.toUpperCase()) {
            case "CLASSIFICATION":
                return CLASSIFICATION;
            case "DETECTION":
                return DETECTION;
            case "CAPTCHA":
                return CAPTCHA;
            case "REGION":
                return REGION;
            case "IMAGE":
                return IMAGE;
            case "COMPARE":
                return COMPARE;
            case "WRITING":
                return WRITING;
            case "SWAG":
                return SWAG;
            case "TAG":
                return TAG;
            case "OLD_CLASSIFICATION":
                return OLD_CLASSIFICATION;
            case "STUDY":
                return STUDY;
            case "DET":
                return DET;
            case "SUPER":
                return SUPER;
            case "LETTERBOX":
                return LETTERBOX;
            case "REGRESSION":
                return REGRESSION;
            case "SEGMENTATION":
                return SEGMENTATION;
            case "INSTANCE":
                return INSTANCE;
            case "ISEG":
                return ISEG;
            default:
                return null;
        }
    }

}
