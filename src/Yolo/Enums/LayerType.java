package Yolo.Enums;

public enum LayerType {

    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    GRU,
    LSTM,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    YOLO,
    ISEG,
    REORG,
    UPSAMPLE,
    LOGXENT,
    L2NORM,
    BLANK;

    public static LayerType getLayerType(String s) {

        String ss = s.replace("[","").replace("]","").toUpperCase();

        switch (ss) {
            case "CONVOLUTIONAL":
                return CONVOLUTIONAL;
            case "DECONVOLUTIONAL":
                return DECONVOLUTIONAL;
            case "CONNECTED":
                return CONNECTED;
            case "MAXPOOL":
                return MAXPOOL;
            case "SOFTMAX":
                return SOFTMAX;
            case "DETECTION":
                return DETECTION;
            case "DROPOUT":
                return DROPOUT;
            case "CROP":
                return CROP;
            case "ROUTE":
                return ROUTE;
            case "COST":
                return COST;
            case "NORMALIZATION":
                return NORMALIZATION;
            case "AVGPOOL":
                return AVGPOOL;
            case "LOCAL":
                return LOCAL;
            case "SHORTCUT":
                return SHORTCUT;
            case "ACTIVE":
                return ACTIVE;
            case "RNN":
                return RNN;
            case "GRU":
                return GRU;
            case "LSTM":
                return LSTM;
            case "CRNN":
                return CRNN;
            case "BATCHNORM":
                return BATCHNORM;
            case "NETWORK":
                return NETWORK;
            case "XNOR":
                return XNOR;
            case "REGION":
                return REGION;
            case "YOLO":
                return YOLO;
            case "ISEG":
                return ISEG;
            case "REORG":
                return REORG;
            case "UPSAMPLE":
                return UPSAMPLE;
            case "LOGXENT":
                return LOGXENT;
            case "L2NORM":
                return L2NORM;
            case "BLANK":
            default:
                return BLANK;
        }
    }

}
