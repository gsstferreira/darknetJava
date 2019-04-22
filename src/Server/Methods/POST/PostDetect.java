package Server.Methods.POST;


import Classes.DetectionResult;
import Classes.Image;
import Server.Handlers.ResponseHandler;
import Yolo.Detector;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.BufferedOutputStream;
import java.io.PrintWriter;
import java.util.Base64;

public abstract class PostDetect {

    private static final Gson gson = new Gson();
    private static final JsonParser jsonParser = new JsonParser();
    private static final String contentJson = "application/json";
    private static final String contentText = "text/html";

    public static void detect(PrintWriter responseOutput, BufferedOutputStream responseData, String data) {

        JsonObject json = (JsonObject)jsonParser.parse(data);

        String b64Im = json.get("imageBytes").getAsString();
        byte[] imageBytes = Base64.getDecoder().decode(b64Im);

        float thresh = 0.5f;

        Image image = Image.loadImageColorMemory(imageBytes,0,0);

        if(image == null) {
            ResponseHandler.responseInternalServerError(responseOutput, responseData, "Could not load request image properly");
        }
        else {
            DetectionResult detections = Detector.runDetectorImage(image,thresh);
            String result = gson.toJson(detections);

            ResponseHandler.responseOk(responseOutput, responseData, result,contentJson);
        }
    }
}
