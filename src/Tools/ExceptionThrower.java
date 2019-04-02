package Tools;

import java.security.InvalidParameterException;

public abstract class ExceptionThrower {

    public static void InvalidParams(String error) {

        throw new InvalidParameterException(error);
    }

}
