package Server.Methods.GET;

import Classes.DetectionResult;
import Server.Handlers.ResponseHandler;
import Yolo.Detector;
import com.google.gson.Gson;

import java.io.*;
import java.security.InvalidParameterException;

public abstract class Method_Detect {

    private static final Gson gson = new Gson();
    private static final String contentJson = "application/json";
    private static final String contentText = "text/html";

    public static void detect(PrintWriter responseOutput, BufferedOutputStream responseData, String params) {

        String[] _params = params.split("&");

        int numPath = 0;
        int numThresh = 0;

        float thresh = 0.5f;
        String path = "";

        for (String s : _params) {
            if (s.contains("thresh")) {
                try {
                    thresh = Float.parseFloat(s.split("=")[1]);
                    numThresh++;

                    if (thresh < 0 || thresh > 1.0f) {
                        throw new InvalidParameterException();
                    }
                } catch (NumberFormatException e) {
                    ResponseHandler.responseInternalServerError(responseOutput, responseData, "'thresh' parameter is not a valid float number");
                    return;
                } catch (InvalidParameterException e) {
                    ResponseHandler.responseInternalServerError(responseOutput, responseData, "'thresh' parameter must be between '0' and '1'");
                    return;
                }

            } else if (s.contains("path")) {
                path = s.split("=")[1].replace("%5C","/");
                numPath++;
            }
        }

        if(numPath == 0) {
            ResponseHandler.responseInternalServerError(responseOutput, responseData, "'path' parameter is required");
        }
        else if(numPath > 1) {
            ResponseHandler.responseInternalServerError(responseOutput, responseData, "Multiple 'path' paramters in query string");
        }
        else if(numThresh > 1) {
            ResponseHandler.responseInternalServerError(responseOutput, responseData, "Multiple 'thresh' paramters in query string");
        }
        else {

            if(new File(path).exists()) {
                DetectionResult detections = Detector.runDetector(path,thresh);
                String result = gson.toJson(detections);

                ResponseHandler.responseOk(responseOutput, responseData, result,contentJson);
            }
            else {
                ResponseHandler.responseInternalServerError(responseOutput, responseData, "'path' parameter does not correspond to a valid file");
            }
        }
    }
}
