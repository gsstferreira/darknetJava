package Yolo.Enums;

public enum ImType {

    PNG,
    BMP,
    TGA,
    JPG;

    public static ImType getImageType(String s) {
        switch (s.toUpperCase()) {
            case "PNG":
                return PNG;
            case "BMP":
                return BMP;
            case "TGA":
                return TGA;
            case "JPG":
            case "JPEG":
                return JPG;
            default:
                return null;
        }
    }
}
