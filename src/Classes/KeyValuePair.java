package Classes;

import Tools.Util;

public class KeyValuePair {

    public final String key;
    public final String value;
    public boolean used;

    public KeyValuePair(String k, String v) {

        key = Util.strip(k);
        value = Util.strip(v);
        used = false;
    }
}
